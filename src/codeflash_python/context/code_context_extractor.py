from __future__ import annotations

import ast
import hashlib
import logging
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash_core.models import FunctionToOptimize  # noqa: TC001
from codeflash_python.code_utils.code_utils import encoded_tokens_len
from codeflash_python.code_utils.config_consts import (
    OPTIMIZATION_CONTEXT_TOKEN_LIMIT,
    READ_WRITABLE_LIMIT_ERROR,
    TESTGEN_CONTEXT_TOKEN_LIMIT,
    TESTGEN_LIMIT_ERROR,
)
from codeflash_python.context.ast_helpers import collect_existing_class_names, parse_and_collect_imports
from codeflash_python.context.class_extraction import enrich_testgen_context
from codeflash_python.context.cst_pruning import parse_code_and_prune_cst
from codeflash_python.context.jedi_helpers import (
    get_function_sources_from_jedi,
    get_function_to_optimize_as_function_source,
)
from codeflash_python.context.type_extraction import extract_parameter_type_constructors
from codeflash_python.context.types import CodeContextType
from codeflash_python.context.unused_definition_remover import remove_unused_definitions_by_function_names
from codeflash_python.models.models import CodeOptimizationContext, CodeString, CodeStringsMarkdown, FunctionSource
from codeflash_python.static_analysis.code_extractor import find_preexisting_objects
from codeflash_python.static_analysis.import_analysis import add_needed_imports_from_module

if TYPE_CHECKING:
    from codeflash_python.context.types import DependencyResolver

logger = logging.getLogger("codeflash_python")


def build_testgen_context(
    helpers_of_fto_dict: dict[Path, set[FunctionSource]],
    helpers_of_helpers_dict: dict[Path, set[FunctionSource]],
    project_root_path: Path,
    *,
    remove_docstrings: bool = False,
    include_enrichment: bool = True,
    function_to_optimize: FunctionToOptimize | None = None,
) -> CodeStringsMarkdown:
    testgen_context = extract_code_markdown_context_from_files(
        helpers_of_fto_dict,
        helpers_of_helpers_dict,
        project_root_path,
        remove_docstrings=remove_docstrings,
        code_context_type=CodeContextType.TESTGEN,
    )

    if include_enrichment:
        enrichment = enrich_testgen_context(testgen_context, project_root_path)
        if enrichment.code_strings:
            testgen_context = CodeStringsMarkdown(code_strings=testgen_context.code_strings + enrichment.code_strings)

        if function_to_optimize is not None:
            result = parse_and_collect_imports(testgen_context)
            existing_classes = collect_existing_class_names(result[0]) if result else set()
            constructor_stubs = extract_parameter_type_constructors(
                function_to_optimize, project_root_path, existing_classes
            )
            if constructor_stubs.code_strings:
                testgen_context = CodeStringsMarkdown(
                    code_strings=testgen_context.code_strings + constructor_stubs.code_strings
                )

    return testgen_context


def get_code_optimization_context(
    function_to_optimize: FunctionToOptimize,
    project_root_path: Path,
    optim_token_limit: int = OPTIMIZATION_CONTEXT_TOKEN_LIMIT,
    testgen_token_limit: int = TESTGEN_CONTEXT_TOKEN_LIMIT,
    call_graph: DependencyResolver | None = None,
) -> CodeOptimizationContext:
    # Get FunctionSource representation of helpers of FTO
    fto_input = {function_to_optimize.file_path: {function_to_optimize.qualified_name}}
    if call_graph is not None:
        helpers_of_fto_dict, helpers_of_fto_list = call_graph.get_callees(fto_input)
    else:
        helpers_of_fto_dict, helpers_of_fto_list = get_function_sources_from_jedi(fto_input, project_root_path)

    # Add function to optimize into helpers of FTO dict, as they'll be processed together
    fto_as_function_source = get_function_to_optimize_as_function_source(function_to_optimize, project_root_path)
    helpers_of_fto_dict[function_to_optimize.file_path].add(fto_as_function_source)

    # Format data to search for helpers of helpers using get_function_sources_from_jedi
    helpers_of_fto_qualified_names_dict = {
        file_path: {source.qualified_name for source in sources} for file_path, sources in helpers_of_fto_dict.items()
    }

    # __init__ functions are automatically considered as helpers of FTO, so we add them to the dict (regardless of whether they exist)
    # This helps us to search for helpers of __init__ functions of classes that contain helpers of FTO
    for qualified_names in helpers_of_fto_qualified_names_dict.values():
        qualified_names.update({f"{qn.rsplit('.', 1)[0]}.__init__" for qn in qualified_names if "." in qn})

    helpers_of_helpers_dict, helpers_of_helpers_list = get_function_sources_from_jedi(
        helpers_of_fto_qualified_names_dict, project_root_path
    )

    # Extract code context for optimization
    final_read_writable_code = extract_code_markdown_context_from_files(
        helpers_of_fto_dict,
        {},
        project_root_path,
        remove_docstrings=False,
        code_context_type=CodeContextType.READ_WRITABLE,
    )

    # Ensure the target file is first in the code blocks so the LLM knows which file to optimize
    target_relative = function_to_optimize.file_path.resolve().relative_to(project_root_path.resolve())
    target_blocks = [cs for cs in final_read_writable_code.code_strings if cs.file_path == target_relative]
    other_blocks = [cs for cs in final_read_writable_code.code_strings if cs.file_path != target_relative]
    if target_blocks:
        final_read_writable_code.code_strings = target_blocks + other_blocks

    read_only_code_markdown = extract_code_markdown_context_from_files(
        helpers_of_fto_dict,
        helpers_of_helpers_dict,
        project_root_path,
        remove_docstrings=False,
        code_context_type=CodeContextType.READ_ONLY,
    )
    hashing_code_context = extract_code_markdown_context_from_files(
        helpers_of_fto_dict,
        helpers_of_helpers_dict,
        project_root_path,
        remove_docstrings=True,
        code_context_type=CodeContextType.HASHING,
    )

    # Handle token limits
    final_read_writable_tokens = encoded_tokens_len(final_read_writable_code.markdown)
    if final_read_writable_tokens > optim_token_limit:
        raise ValueError(READ_WRITABLE_LIMIT_ERROR)

    # Setup preexisting objects for code replacer
    preexisting_objects = set(
        chain(
            *(find_preexisting_objects(codestring.code) for codestring in final_read_writable_code.code_strings),
            *(find_preexisting_objects(codestring.code) for codestring in read_only_code_markdown.code_strings),
        )
    )
    read_only_context_code = read_only_code_markdown.markdown

    # Progressive fallback for read-only context token limits
    read_only_tokens = encoded_tokens_len(read_only_context_code)
    if final_read_writable_tokens + read_only_tokens > optim_token_limit:
        logger.debug("Code context has exceeded token limit, removing docstrings from read-only code")
        read_only_code_no_docstrings = extract_code_markdown_context_from_files(
            helpers_of_fto_dict, helpers_of_helpers_dict, project_root_path, remove_docstrings=True
        )
        read_only_context_code = read_only_code_no_docstrings.markdown
        if final_read_writable_tokens + encoded_tokens_len(read_only_context_code) > optim_token_limit:
            logger.debug("Code context has exceeded token limit, removing read-only code")
            read_only_context_code = ""

    # Progressive fallback for testgen context token limits
    testgen_context = build_testgen_context(
        helpers_of_fto_dict, helpers_of_helpers_dict, project_root_path, function_to_optimize=function_to_optimize
    )

    if encoded_tokens_len(testgen_context.markdown) > testgen_token_limit:
        logger.debug("Testgen context exceeded token limit, removing docstrings")
        testgen_context = build_testgen_context(
            helpers_of_fto_dict,
            helpers_of_helpers_dict,
            project_root_path,
            remove_docstrings=True,
            function_to_optimize=function_to_optimize,
        )

        if encoded_tokens_len(testgen_context.markdown) > testgen_token_limit:
            logger.debug("Testgen context still exceeded token limit, removing enrichment")
            testgen_context = build_testgen_context(
                helpers_of_fto_dict,
                helpers_of_helpers_dict,
                project_root_path,
                remove_docstrings=True,
                include_enrichment=False,
            )

            if encoded_tokens_len(testgen_context.markdown) > testgen_token_limit:
                raise ValueError(TESTGEN_LIMIT_ERROR)
    code_hash_context = hashing_code_context.markdown
    code_hash = hashlib.sha256(code_hash_context.encode("utf-8")).hexdigest()

    all_helper_fqns = list({fs.fully_qualified_name for fs in helpers_of_fto_list + helpers_of_helpers_list})

    return CodeOptimizationContext(
        testgen_context=testgen_context,
        read_writable_code=final_read_writable_code,
        read_only_context_code=read_only_context_code,
        hashing_code_context=code_hash_context,
        hashing_code_context_hash=code_hash,
        helper_functions=helpers_of_fto_list,
        testgen_helper_fqns=all_helper_fqns,
        preexisting_objects=preexisting_objects,
    )


def process_file_context(
    file_path: Path,
    primary_qualified_names: set[str],
    secondary_qualified_names: set[str],
    code_context_type: CodeContextType,
    remove_docstrings: bool,
    project_root_path: Path,
    helper_functions: list[FunctionSource],
) -> CodeString | None:
    try:
        original_code = file_path.read_text("utf8")
    except Exception as e:
        logger.exception("Error while parsing %s: %s", file_path, e)
        return None

    try:
        all_names = primary_qualified_names | secondary_qualified_names
        code_without_unused_defs = remove_unused_definitions_by_function_names(original_code, all_names)
        pruned_module = parse_code_and_prune_cst(
            code_without_unused_defs,
            code_context_type,
            primary_qualified_names,
            secondary_qualified_names,
            remove_docstrings,
        )
    except ValueError as e:
        logger.debug("Error while getting read-only code: %s", e)
        return None

    if pruned_module.code.strip():
        if code_context_type == CodeContextType.HASHING:
            code_context = ast.unparse(ast.parse(pruned_module.code))
        else:
            code_context = add_needed_imports_from_module(
                src_module_code=original_code,
                dst_module_code=pruned_module,
                src_path=file_path,
                dst_path=file_path,
                project_root=project_root_path,
                helper_functions=helper_functions,
            )
        try:
            relative_path = file_path.resolve().relative_to(project_root_path.resolve())
        except ValueError:
            relative_path = file_path
        return CodeString(code=code_context, file_path=relative_path)
    return None


def extract_code_markdown_context_from_files(
    helpers_of_fto: dict[Path, set[FunctionSource]],
    helpers_of_helpers: dict[Path, set[FunctionSource]],
    project_root_path: Path,
    remove_docstrings: bool = False,
    code_context_type: CodeContextType = CodeContextType.READ_ONLY,
) -> CodeStringsMarkdown:
    """Extract code context from files containing target functions and their helpers, formatting them as markdown.

    This function processes two sets of files:
    1. Files containing the function to optimize (fto) and their first-degree helpers
    2. Files containing only helpers of helpers (with no overlap with the first set)

    For each file, it extracts relevant code based on the specified context type, adds necessary
    imports, and combines them into a structured markdown format.

    Args:
    ----
        helpers_of_fto: Dictionary mapping file paths to sets of Function Sources of function to optimize and its helpers
        helpers_of_helpers: Dictionary mapping file paths to sets of Function Sources of helpers of helper functions
        project_root_path: Root path of the project
        remove_docstrings: Whether to remove docstrings from the extracted code
        code_context_type: Type of code context to extract (READ_ONLY, READ_WRITABLE, or TESTGEN)

    Returns:
    -------
        CodeStringsMarkdown containing the extracted code context with necessary imports,
        formatted for inclusion in markdown

    """
    # Rearrange to remove overlaps, so we only access each file path once
    helpers_of_helpers_no_overlap = defaultdict(set)
    for file_path, function_sources in helpers_of_helpers.items():
        if file_path in helpers_of_fto:
            # Remove duplicates within the same file path, in case a helper of helper is also a helper of fto
            helpers_of_helpers[file_path] -= helpers_of_fto[file_path]
        else:
            helpers_of_helpers_no_overlap[file_path] = function_sources
    code_context_markdown = CodeStringsMarkdown()
    # Extract code from file paths that contain fto and first degree helpers. helpers of helpers may also be included if they are in the same files
    for file_path, function_sources in helpers_of_fto.items():
        qualified_function_names = {func.qualified_name for func in function_sources}
        helpers_of_helpers_qualified_names = {func.qualified_name for func in helpers_of_helpers.get(file_path, set())}
        helper_functions = list(helpers_of_fto.get(file_path, set()) | helpers_of_helpers.get(file_path, set()))

        result = process_file_context(
            file_path=file_path,
            primary_qualified_names=qualified_function_names,
            secondary_qualified_names=helpers_of_helpers_qualified_names,
            code_context_type=code_context_type,
            remove_docstrings=remove_docstrings,
            project_root_path=project_root_path,
            helper_functions=helper_functions,
        )

        if result is not None:
            code_context_markdown.code_strings.append(result)
    # Extract code from file paths containing helpers of helpers
    for file_path, helper_function_sources in helpers_of_helpers_no_overlap.items():
        qualified_helper_function_names = {func.qualified_name for func in helper_function_sources}
        helper_functions = list(helpers_of_helpers_no_overlap.get(file_path, set()))

        result = process_file_context(
            file_path=file_path,
            primary_qualified_names=set(),
            secondary_qualified_names=qualified_helper_function_names,
            code_context_type=code_context_type,
            remove_docstrings=remove_docstrings,
            project_root_path=project_root_path,
            helper_functions=helper_functions,
        )

        if result is not None:
            code_context_markdown.code_strings.append(result)
    return code_context_markdown
