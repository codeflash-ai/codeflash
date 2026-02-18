from __future__ import annotations

import ast
import hashlib
import os
from collections import defaultdict, deque
from itertools import chain
from typing import TYPE_CHECKING

import libcst as cst

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_extractor import add_needed_imports_from_module, find_preexisting_objects
from codeflash.code_utils.code_utils import encoded_tokens_len, get_qualified_name, path_belongs_to_site_packages
from codeflash.code_utils.config_consts import OPTIMIZATION_CONTEXT_TOKEN_LIMIT, TESTGEN_CONTEXT_TOKEN_LIMIT
from codeflash.discovery.functions_to_optimize import FunctionToOptimize  # noqa: TC001

# Language support imports for multi-language code context extraction
from codeflash.languages import Language, is_python
from codeflash.languages.python.context.unused_definition_remover import (
    collect_top_level_defs_with_usages,
    get_section_names,
    is_assignment_used,
    recurse_sections,
    remove_unused_definitions_by_function_names,
)
from codeflash.models.models import (
    CodeContextType,
    CodeOptimizationContext,
    CodeString,
    CodeStringsMarkdown,
    FunctionSource,
)
from codeflash.optimization.function_context import belongs_to_function_qualified

if TYPE_CHECKING:
    from pathlib import Path

    from jedi.api.classes import Name

    from codeflash.languages.base import HelperFunction
    from codeflash.languages.python.context.unused_definition_remover import UsageInfo

# Error message constants
READ_WRITABLE_LIMIT_ERROR = "Read-writable code has exceeded token limit, cannot proceed"
TESTGEN_LIMIT_ERROR = "Testgen code context has exceeded token limit, cannot proceed"


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

        type_context_strings: list[CodeString] = []
        if function_to_optimize is not None:
            result = _parse_and_collect_imports(testgen_context)
            existing_classes = collect_existing_class_names(result[0]) if result else set()
            type_context = extract_type_context_for_testgen(function_to_optimize, project_root_path, existing_classes)
            if type_context.code_strings:
                testgen_context = CodeStringsMarkdown(
                    code_strings=testgen_context.code_strings + type_context.code_strings
                )
                type_context_strings = type_context.code_strings

        # Enrich field types from all newly extracted classes (enrichment + type context)
        new_classes = CodeStringsMarkdown(code_strings=enrichment.code_strings + type_context_strings)
        if new_classes.code_strings:
            updated_result = _parse_and_collect_imports(testgen_context)
            updated_existing = collect_existing_class_names(updated_result[0]) if updated_result else set()
            field_type_enrichment = enrich_type_context_classes(new_classes, updated_existing, project_root_path)
            if field_type_enrichment.code_strings:
                testgen_context = CodeStringsMarkdown(
                    code_strings=testgen_context.code_strings + field_type_enrichment.code_strings
                )

    return testgen_context


def get_code_optimization_context(
    function_to_optimize: FunctionToOptimize,
    project_root_path: Path,
    optim_token_limit: int = OPTIMIZATION_CONTEXT_TOKEN_LIMIT,
    testgen_token_limit: int = TESTGEN_CONTEXT_TOKEN_LIMIT,
) -> CodeOptimizationContext:
    # Route to language-specific implementation for non-Python languages
    if not is_python():
        return get_code_optimization_context_for_language(
            function_to_optimize, project_root_path, optim_token_limit, testgen_token_limit
        )

    # Get FunctionSource representation of helpers of FTO
    helpers_of_fto_dict, helpers_of_fto_list = get_function_sources_from_jedi(
        {function_to_optimize.file_path: {function_to_optimize.qualified_name}}, project_root_path
    )

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

    # Get FunctionSource representation of helpers of helpers of FTO
    helpers_of_helpers_dict, _helpers_of_helpers_list = get_function_sources_from_jedi(
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

    return CodeOptimizationContext(
        testgen_context=testgen_context,
        read_writable_code=final_read_writable_code,
        read_only_context_code=read_only_context_code,
        hashing_code_context=code_hash_context,
        hashing_code_context_hash=code_hash,
        helper_functions=helpers_of_fto_list,
        preexisting_objects=preexisting_objects,
    )


def get_code_optimization_context_for_language(
    function_to_optimize: FunctionToOptimize,
    project_root_path: Path,
    optim_token_limit: int = OPTIMIZATION_CONTEXT_TOKEN_LIMIT,
    testgen_token_limit: int = TESTGEN_CONTEXT_TOKEN_LIMIT,
) -> CodeOptimizationContext:
    """Extract code optimization context for non-Python languages.

    Uses the language support abstraction to extract code context and converts
    it to the CodeOptimizationContext format expected by the pipeline.

    This function supports multi-file context extraction, grouping helpers by file
    and creating proper CodeStringsMarkdown with file paths for multi-file replacement.

    Args:
        function_to_optimize: The function to extract context for.
        project_root_path: Root of the project.
        optim_token_limit: Token limit for optimization context.
        testgen_token_limit: Token limit for testgen context.

    Returns:
        CodeOptimizationContext with target code and dependencies.

    """
    from codeflash.languages import get_language_support

    # Get language support for this function
    language = Language(function_to_optimize.language)
    lang_support = get_language_support(language)

    # Extract code context using language support
    code_context = lang_support.extract_code_context(function_to_optimize, project_root_path, project_root_path)

    # Build imports string if available
    imports_code = "\n".join(code_context.imports) if code_context.imports else ""

    # Get relative path for target file
    try:
        target_relative_path = function_to_optimize.file_path.resolve().relative_to(project_root_path.resolve())
    except ValueError:
        target_relative_path = function_to_optimize.file_path

    # Group helpers by file path
    helpers_by_file: dict[Path, list[HelperFunction]] = defaultdict(list)
    helper_function_sources = []

    for helper in code_context.helper_functions:
        helpers_by_file[helper.file_path].append(helper)

        # Convert to FunctionSource for pipeline compatibility
        helper_function_sources.append(
            FunctionSource(
                file_path=helper.file_path,
                qualified_name=helper.qualified_name,
                fully_qualified_name=helper.qualified_name,
                only_function_name=helper.name,
                source_code=helper.source_code,
                jedi_definition=None,
            )
        )

    # Build read-writable code (target file + same-file helpers + global variables)
    read_writable_code_strings = []

    # Combine target code with same-file helpers
    target_file_code = code_context.target_code
    same_file_helpers = helpers_by_file.get(function_to_optimize.file_path, [])
    if same_file_helpers:
        helper_code = "\n\n".join(h.source_code for h in same_file_helpers)
        target_file_code = target_file_code + "\n\n" + helper_code

    # Note: code_context.read_only_context contains type definitions and global variables
    # These should be passed as read-only context to the AI, not prepended to the target code
    # If prepended to target code, the AI treats them as code to optimize and includes them in output

    # Add imports to target file code
    if imports_code:
        target_file_code = imports_code + "\n\n" + target_file_code

    read_writable_code_strings.append(
        CodeString(code=target_file_code, file_path=target_relative_path, language=function_to_optimize.language)
    )

    # Add helper files (cross-file helpers)
    for file_path, file_helpers in helpers_by_file.items():
        if file_path == function_to_optimize.file_path:
            continue  # Already included in target file

        try:
            helper_relative_path = file_path.resolve().relative_to(project_root_path.resolve())
        except ValueError:
            helper_relative_path = file_path

        # Combine all helpers from this file
        combined_helper_code = "\n\n".join(h.source_code for h in file_helpers)

        read_writable_code_strings.append(
            CodeString(
                code=combined_helper_code, file_path=helper_relative_path, language=function_to_optimize.language
            )
        )

    read_writable_code = CodeStringsMarkdown(
        code_strings=read_writable_code_strings, language=function_to_optimize.language
    )

    # Build testgen context (same as read_writable for non-Python)
    testgen_context = CodeStringsMarkdown(
        code_strings=read_writable_code_strings.copy(), language=function_to_optimize.language
    )

    # Check token limits
    read_writable_tokens = encoded_tokens_len(read_writable_code.markdown)
    if read_writable_tokens > optim_token_limit:
        raise ValueError(READ_WRITABLE_LIMIT_ERROR)

    testgen_tokens = encoded_tokens_len(testgen_context.markdown)
    if testgen_tokens > testgen_token_limit:
        raise ValueError(TESTGEN_LIMIT_ERROR)

    # Generate code hash from all read-writable code
    code_hash = hashlib.sha256(read_writable_code.flat.encode("utf-8")).hexdigest()

    return CodeOptimizationContext(
        testgen_context=testgen_context,
        read_writable_code=read_writable_code,
        # Pass type definitions and globals as read-only context for the AI
        # This way the AI sees them as context but doesn't include them in optimized output
        read_only_context_code=code_context.read_only_context,
        hashing_code_context=read_writable_code.flat,
        hashing_code_context_hash=code_hash,
        helper_functions=helper_function_sources,
        preexisting_objects=set(),  # Not implemented for non-Python yet
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
        logger.exception(f"Error while parsing {file_path}: {e}")
        return None

    try:
        all_names = primary_qualified_names | secondary_qualified_names
        code_without_unused_defs = remove_unused_definitions_by_function_names(original_code, all_names)
        code_context = parse_code_and_prune_cst(
            code_without_unused_defs,
            code_context_type,
            primary_qualified_names,
            secondary_qualified_names,
            remove_docstrings,
        )
    except ValueError as e:
        logger.debug(f"Error while getting read-only code: {e}")
        return None

    if code_context.strip():
        if code_context_type != CodeContextType.HASHING:
            code_context = add_needed_imports_from_module(
                src_module_code=original_code,
                dst_module_code=code_context,
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


def get_function_to_optimize_as_function_source(
    function_to_optimize: FunctionToOptimize, project_root_path: Path
) -> FunctionSource:
    import jedi

    # Use jedi to find function to optimize
    script = jedi.Script(path=function_to_optimize.file_path, project=jedi.Project(path=project_root_path))

    # Get all names in the file
    names = script.get_names(all_scopes=True, definitions=True, references=False)

    # Find the name that matches our function
    for name in names:
        try:
            if (
                name.type == "function"
                and name.full_name
                and name.name == function_to_optimize.function_name
                and name.full_name.startswith(name.module_name)
                and get_qualified_name(name.module_name, name.full_name) == function_to_optimize.qualified_name
            ):
                return FunctionSource(
                    file_path=function_to_optimize.file_path,
                    qualified_name=function_to_optimize.qualified_name,
                    fully_qualified_name=name.full_name,
                    only_function_name=name.name,
                    source_code=name.get_line_code(),
                    jedi_definition=name,
                )
        except Exception as e:
            logger.exception(f"Error while getting function source: {e}")
            continue
    raise ValueError(
        f"Could not find function {function_to_optimize.function_name} in {function_to_optimize.file_path}"  # noqa: EM102
    )


def get_function_sources_from_jedi(
    file_path_to_qualified_function_names: dict[Path, set[str]], project_root_path: Path
) -> tuple[dict[Path, set[FunctionSource]], list[FunctionSource]]:
    import jedi

    file_path_to_function_source = defaultdict(set)
    function_source_list: list[FunctionSource] = []
    for file_path, qualified_function_names in file_path_to_qualified_function_names.items():
        script = jedi.Script(path=file_path, project=jedi.Project(path=project_root_path))
        file_refs = script.get_names(all_scopes=True, definitions=False, references=True)

        for qualified_function_name in qualified_function_names:
            names = [
                ref
                for ref in file_refs
                if ref.full_name and belongs_to_function_qualified(ref, qualified_function_name)
            ]
            for name in names:
                try:
                    definitions: list[Name] = name.goto(follow_imports=True, follow_builtin_imports=False)
                except Exception:
                    logger.debug(f"Error while getting definitions for {qualified_function_name}")
                    definitions = []
                if definitions:
                    # TODO: there can be multiple definitions, see how to handle such cases
                    definition = definitions[0]
                    definition_path = definition.module_path

                    # The definition is part of this project and not defined within the original function
                    is_valid_definition = (
                        definition_path is not None
                        and not path_belongs_to_site_packages(definition_path)
                        and str(definition_path).startswith(str(project_root_path) + os.sep)
                        and definition.full_name
                        and not belongs_to_function_qualified(definition, qualified_function_name)
                        and definition.full_name.startswith(definition.module_name)
                    )
                    if is_valid_definition and definition.type in ("function", "class"):
                        if definition.type == "function":
                            fqn = definition.full_name
                            func_name = definition.name
                        else:
                            # When a class is instantiated (e.g., MyClass()), track its __init__ as a helper
                            # This ensures the class definition with constructor is included in testgen context
                            fqn = f"{definition.full_name}.__init__"
                            func_name = "__init__"
                        qualified_name = get_qualified_name(definition.module_name, fqn)
                        # Avoid nested functions or classes. Only class.function is allowed
                        if len(qualified_name.split(".")) <= 2:
                            function_source = FunctionSource(
                                file_path=definition_path,
                                qualified_name=qualified_name,
                                fully_qualified_name=fqn,
                                only_function_name=func_name,
                                source_code=definition.get_line_code(),
                                jedi_definition=definition,
                            )
                            file_path_to_function_source[definition_path].add(function_source)
                            function_source_list.append(function_source)

    return file_path_to_function_source, function_source_list


def _parse_and_collect_imports(code_context: CodeStringsMarkdown) -> tuple[ast.Module, dict[str, str]] | None:
    all_code = "\n".join(cs.code for cs in code_context.code_strings)
    try:
        tree = ast.parse(all_code)
    except SyntaxError:
        return None
    imported_names: dict[str, str] = {}

    # Directly iterate over the module body and nested structures instead of ast.walk
    # This avoids traversing every single node in the tree
    def collect_imports(nodes: list[ast.stmt]) -> None:
        for node in nodes:
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    if alias.name != "*":
                        imported_name = alias.asname if alias.asname else alias.name
                        imported_names[imported_name] = node.module
            # Recursively check nested structures (function defs, class defs, if statements, etc.)
            elif isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.If,
                    ast.For,
                    ast.AsyncFor,
                    ast.While,
                    ast.With,
                    ast.AsyncWith,
                    ast.Try,
                    ast.ExceptHandler,
                ),
            ):
                if hasattr(node, "body"):
                    collect_imports(node.body)
                if hasattr(node, "orelse"):
                    collect_imports(node.orelse)
                if hasattr(node, "finalbody"):
                    collect_imports(node.finalbody)
                if hasattr(node, "handlers"):
                    for handler in node.handlers:
                        collect_imports(handler.body)
            # Handle match/case statements (Python 3.10+)
            elif hasattr(ast, "Match") and isinstance(node, ast.Match):
                for case in node.cases:
                    collect_imports(case.body)

    collect_imports(tree.body)
    return tree, imported_names


def collect_existing_class_names(tree: ast.Module) -> set[str]:
    class_names = set()
    stack = list(tree.body)

    while stack:
        node = stack.pop()
        if isinstance(node, ast.ClassDef):
            class_names.add(node.name)
            stack.extend(node.body)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stack.extend(node.body)
        elif isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
            stack.extend(node.body)
            if hasattr(node, "orelse"):
                stack.extend(node.orelse)
        elif isinstance(node, ast.Try):
            stack.extend(node.body)
            stack.extend(node.orelse)
            stack.extend(node.finalbody)
            for handler in node.handlers:
                stack.extend(handler.body)

    return class_names


BUILTIN_AND_TYPING_NAMES = frozenset(
    {
        "int",
        "str",
        "float",
        "bool",
        "bytes",
        "bytearray",
        "complex",
        "list",
        "dict",
        "set",
        "frozenset",
        "tuple",
        "type",
        "object",
        "None",
        "NoneType",
        "Ellipsis",
        "NotImplemented",
        "memoryview",
        "range",
        "slice",
        "property",
        "classmethod",
        "staticmethod",
        "super",
        "Optional",
        "Union",
        "Any",
        "List",
        "Dict",
        "Set",
        "FrozenSet",
        "Tuple",
        "Type",
        "Callable",
        "Iterator",
        "Generator",
        "Coroutine",
        "AsyncGenerator",
        "AsyncIterator",
        "Iterable",
        "AsyncIterable",
        "Sequence",
        "MutableSequence",
        "Mapping",
        "MutableMapping",
        "Collection",
        "Awaitable",
        "Literal",
        "Final",
        "ClassVar",
        "TypeVar",
        "TypeAlias",
        "ParamSpec",
        "Concatenate",
        "Annotated",
        "TypeGuard",
        "Self",
        "Unpack",
        "TypeVarTuple",
        "Never",
        "NoReturn",
        "SupportsInt",
        "SupportsFloat",
        "SupportsComplex",
        "SupportsBytes",
        "SupportsAbs",
        "SupportsRound",
        "IO",
        "TextIO",
        "BinaryIO",
        "Pattern",
        "Match",
    }
)


def collect_type_names_from_annotation(node: ast.expr | None) -> set[str]:
    if node is None:
        return set()
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Subscript):
        names = collect_type_names_from_annotation(node.value)
        names |= collect_type_names_from_annotation(node.slice)
        return names
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return collect_type_names_from_annotation(node.left) | collect_type_names_from_annotation(node.right)
    if isinstance(node, ast.Tuple):
        names = set[str]()
        for elt in node.elts:
            names |= collect_type_names_from_annotation(elt)
        return names
    return set()


def collect_names_from_function_body(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                names.add(node.func.value.id)
            if isinstance(node.func, ast.Name) and node.func.id in ("isinstance", "issubclass") and node.args:
                second_arg = node.args[1] if len(node.args) > 1 else None
                if isinstance(second_arg, ast.Name):
                    names.add(second_arg.id)
                elif isinstance(second_arg, ast.Tuple):
                    for elt in second_arg.elts:
                        if isinstance(elt, ast.Name):
                            names.add(elt.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            names.add(node.value.id)
    return names


def extract_full_class_from_module(class_name: str, module_source: str, module_tree: ast.Module) -> str | None:
    class_node = None
    # Use a deque-based BFS to find the first matching ClassDef (preserves ast.walk order)
    q: deque[ast.AST] = deque([module_tree])
    while q:
        candidate = q.popleft()
        if isinstance(candidate, ast.ClassDef) and candidate.name == class_name:
            class_node = candidate
            break
        q.extend(ast.iter_child_nodes(candidate))

    if class_node is None:
        return None

    lines = module_source.split("\n")
    start_line = class_node.lineno
    if class_node.decorator_list:
        start_line = min(d.lineno for d in class_node.decorator_list)
    return "\n".join(lines[start_line - 1 : class_node.end_lineno])


def extract_init_stub_from_class(class_name: str, module_source: str, module_tree: ast.Module) -> str | None:
    class_node = None
    # Use a deque-based BFS to find the first matching ClassDef (preserves ast.walk order)
    q: deque[ast.AST] = deque([module_tree])
    while q:
        candidate = q.popleft()
        if isinstance(candidate, ast.ClassDef) and candidate.name == class_name:
            class_node = candidate
            break
        q.extend(ast.iter_child_nodes(candidate))

    if class_node is None:
        return None

    lines = module_source.splitlines()
    relevant_nodes: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_relevant = False
            if item.name in ("__init__", "__post_init__"):
                is_relevant = True
            else:
                # Check decorators explicitly to avoid generator overhead
                for d in item.decorator_list:
                    if (isinstance(d, ast.Name) and d.id == "property") or (
                        isinstance(d, ast.Attribute) and d.attr == "property"
                    ):
                        is_relevant = True
                        break
            if is_relevant:
                relevant_nodes.append(item)

    if not relevant_nodes:
        return None

    snippets: list[str] = []
    for fn_node in relevant_nodes:
        start = fn_node.lineno
        if fn_node.decorator_list:
            # Compute minimum decorator lineno with an explicit loop (avoids generator/min overhead)
            m = start
            for d in fn_node.decorator_list:
                m = min(m, d.lineno)
            start = m
        snippets.append("\n".join(lines[start - 1 : fn_node.end_lineno]))

    return f"class {class_name}:\n" + "\n".join(snippets)


def extract_type_context_for_testgen(
    function_to_optimize: FunctionToOptimize, project_root_path: Path, existing_class_names: set[str]
) -> CodeStringsMarkdown:
    import jedi

    try:
        source = function_to_optimize.file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return CodeStringsMarkdown(code_strings=[])

    func_node = None
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_to_optimize.function_name
        ):
            if function_to_optimize.starting_line is not None and node.lineno != function_to_optimize.starting_line:
                continue
            func_node = node
            break
    if func_node is None:
        return CodeStringsMarkdown(code_strings=[])

    type_names: set[str] = set()
    for arg in func_node.args.args + func_node.args.posonlyargs + func_node.args.kwonlyargs:
        type_names |= collect_type_names_from_annotation(arg.annotation)
    if func_node.args.vararg:
        type_names |= collect_type_names_from_annotation(func_node.args.vararg.annotation)
    if func_node.args.kwarg:
        type_names |= collect_type_names_from_annotation(func_node.args.kwarg.annotation)

    type_names |= collect_names_from_function_body(func_node)

    type_names -= BUILTIN_AND_TYPING_NAMES
    type_names -= existing_class_names
    if not type_names:
        return CodeStringsMarkdown(code_strings=[])

    import_map: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                import_map[name] = node.module

    code_strings: list[CodeString] = []
    module_cache: dict[Path, tuple[str, ast.Module]] = {}

    for type_name in sorted(type_names):
        module_name = import_map.get(type_name)
        if not module_name:
            continue
        try:
            script_code = f"from {module_name} import {type_name}"
            script = jedi.Script(script_code, project=jedi.Project(path=project_root_path))
            definitions = script.goto(1, len(f"from {module_name} import ") + len(type_name), follow_imports=True)
            if not definitions:
                continue

            module_path = definitions[0].module_path
            if not module_path:
                continue

            resolved_module = module_path.resolve()
            module_str = str(resolved_module)
            is_project = module_str.startswith(str(project_root_path.resolve()))
            is_third_party = "site-packages" in module_str
            if not is_project and not is_third_party:
                continue

            if module_path in module_cache:
                mod_source, mod_tree = module_cache[module_path]
            else:
                mod_source = module_path.read_text(encoding="utf-8")
                mod_tree = ast.parse(mod_source)
                module_cache[module_path] = (mod_source, mod_tree)

            class_source = extract_full_class_from_module(type_name, mod_source, mod_tree)
            if class_source is None:
                resolved_class = resolve_instance_class_name(type_name, mod_tree)
                if resolved_class and resolved_class not in existing_class_names:
                    class_source = extract_full_class_from_module(resolved_class, mod_source, mod_tree)
            if class_source:
                code_strings.append(CodeString(code=class_source, file_path=module_path))
        except Exception:
            logger.debug(f"Error extracting type context for {type_name} from {module_name}")
            continue

    return CodeStringsMarkdown(code_strings=code_strings)


def resolve_instance_class_name(name: str, module_tree: ast.Module) -> str | None:
    for node in module_tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    value = node.value
                    if isinstance(value, ast.Call):
                        func = value.func
                        if isinstance(func, ast.Name):
                            return func.id
                        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                            return func.value.id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            ann = node.annotation
            if isinstance(ann, ast.Name):
                return ann.id
            if isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name):
                return ann.value.id
    return None


def enrich_testgen_context(code_context: CodeStringsMarkdown, project_root_path: Path) -> CodeStringsMarkdown:
    import jedi

    result = _parse_and_collect_imports(code_context)
    if result is None:
        return CodeStringsMarkdown(code_strings=[])
    tree, imported_names = result

    if not imported_names:
        return CodeStringsMarkdown(code_strings=[])

    existing_classes = collect_existing_class_names(tree)

    code_strings: list[CodeString] = []
    emitted_class_names: set[str] = set()

    # --- Step 1: Project class definitions (jedi resolution + recursive base extraction) ---
    extracted_classes: set[tuple[Path, str]] = set()
    module_cache: dict[Path, tuple[str, ast.Module]] = {}
    module_import_maps: dict[Path, dict[str, str]] = {}

    def get_module_source_and_tree(module_path: Path) -> tuple[str, ast.Module] | None:
        if module_path in module_cache:
            return module_cache[module_path]
        try:
            module_source = module_path.read_text(encoding="utf-8")
            module_tree = ast.parse(module_source)
        except Exception:
            return None
        else:
            module_cache[module_path] = (module_source, module_tree)
            return module_source, module_tree

    def get_module_import_map(module_path: Path, module_tree: ast.Module) -> dict[str, str]:
        if module_path in module_import_maps:
            return module_import_maps[module_path]
        import_map: dict[str, str] = {}
        for node in ast.walk(module_tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    import_map[name] = node.module
        module_import_maps[module_path] = import_map
        return import_map

    def extract_class_and_bases(
        class_name: str, module_path: Path, module_source: str, module_tree: ast.Module, depth: int = 0
    ) -> None:
        if depth >= 3 or (module_path, class_name) in extracted_classes:
            return

        class_node = None
        for node in ast.walk(module_tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                class_node = node
                break

        if class_node is None:
            return

        for base in class_node.bases:
            base_name = None
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                continue

            if not base_name or base_name in existing_classes:
                continue

            # Check if the base class is defined locally in this module
            local_base = any(isinstance(n, ast.ClassDef) and n.name == base_name for n in ast.walk(module_tree))
            if local_base:
                extract_class_and_bases(base_name, module_path, module_source, module_tree, depth + 1)
            else:
                # Resolve cross-module base class via imports
                import_map = get_module_import_map(module_path, module_tree)
                base_module_name = import_map.get(base_name)
                if not base_module_name:
                    continue
                try:
                    script_code = f"from {base_module_name} import {base_name}"
                    script = jedi.Script(script_code, project=jedi.Project(path=project_root_path))
                    definitions = script.goto(
                        1, len(f"from {base_module_name} import ") + len(base_name), follow_imports=True
                    )
                    if not definitions or not definitions[0].module_path:
                        continue
                    base_module_path = definitions[0].module_path
                    resolved_str = str(base_module_path.resolve())
                    is_project = resolved_str.startswith(str(project_root_path.resolve()) + os.sep)
                    is_third_party = "site-packages" in resolved_str
                    if not is_project and not is_third_party:
                        continue
                    base_mod = get_module_source_and_tree(base_module_path)
                    if base_mod is None:
                        continue
                    extract_class_and_bases(base_name, base_module_path, base_mod[0], base_mod[1], depth + 1)
                except Exception:
                    logger.debug(f"Error resolving cross-module base class {base_name} from {base_module_name}")
                    continue

        if (module_path, class_name) in extracted_classes:
            return

        lines = module_source.split("\n")
        start_line = class_node.lineno
        if class_node.decorator_list:
            start_line = min(d.lineno for d in class_node.decorator_list)
        class_source = "\n".join(lines[start_line - 1 : class_node.end_lineno])

        full_source = class_source

        code_strings.append(CodeString(code=full_source, file_path=module_path))
        extracted_classes.add((module_path, class_name))
        emitted_class_names.add(class_name)

    for name, module_name in imported_names.items():
        if name in existing_classes or module_name == "__future__":
            continue
        try:
            test_code = f"import {module_name}"
            script = jedi.Script(test_code, project=jedi.Project(path=project_root_path))
            completions = script.goto(1, len(test_code))

            if not completions:
                continue

            module_path = completions[0].module_path
            if not module_path:
                continue

            resolved_module = module_path.resolve()
            module_str = str(resolved_module)
            is_project = module_str.startswith(str(project_root_path.resolve()) + os.sep)
            is_third_party = "site-packages" in module_str
            if not is_project and not is_third_party:
                continue

            mod_result = get_module_source_and_tree(module_path)
            if mod_result is None:
                continue
            module_source, module_tree = mod_result

            extract_class_and_bases(name, module_path, module_source, module_tree)

            if (module_path, name) not in extracted_classes:
                resolved_class = resolve_instance_class_name(name, module_tree)
                if resolved_class and resolved_class not in existing_classes:
                    extract_class_and_bases(resolved_class, module_path, module_source, module_tree)

        except Exception:
            logger.debug(f"Error extracting class definition for {name} from {module_name}")
            continue

    return CodeStringsMarkdown(code_strings=code_strings)


def enrich_type_context_classes(
    type_context: CodeStringsMarkdown, existing_class_names: set[str], project_root_path: Path
) -> CodeStringsMarkdown:
    import jedi

    code_strings: list[CodeString] = []
    emitted: set[str] = set()
    module_cache: dict[Path, tuple[str, ast.Module]] = {}

    for cs in type_context.code_strings:
        try:
            snippet_tree = ast.parse(cs.code)
        except SyntaxError:
            continue

        # Collect type names from field annotations of extracted classes
        type_names: set[str] = set()
        for node in ast.walk(snippet_tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and item.annotation:
                        type_names |= collect_type_names_from_annotation(item.annotation)

        type_names -= BUILTIN_AND_TYPING_NAMES
        type_names -= existing_class_names
        type_names -= emitted
        if not type_names:
            continue

        # Build import map from the source file, not the snippet
        source_path = cs.file_path
        if not source_path:
            continue
        import_map: dict[str, str] = {}
        if source_path in module_cache:
            source_code, source_tree = module_cache[source_path]
        else:
            try:
                source_code = source_path.read_text(encoding="utf-8")
                source_tree = ast.parse(source_code)
                module_cache[source_path] = (source_code, source_tree)
            except Exception:
                continue
        for snode in ast.walk(source_tree):
            if isinstance(snode, ast.ImportFrom) and snode.module:
                for alias in snode.names:
                    name = alias.asname if alias.asname else alias.name
                    import_map[name] = snode.module

        for type_name in sorted(type_names):
            module_name = import_map.get(type_name)
            if not module_name:
                continue
            try:
                script_code = f"from {module_name} import {type_name}"
                script = jedi.Script(script_code, project=jedi.Project(path=project_root_path))
                definitions = script.goto(1, len(f"from {module_name} import ") + len(type_name), follow_imports=True)
                if not definitions or not definitions[0].module_path:
                    continue

                mod_path = definitions[0].module_path
                resolved_str = str(mod_path.resolve())
                is_project = resolved_str.startswith(str(project_root_path.resolve()))
                is_third_party = "site-packages" in resolved_str
                if not is_project and not is_third_party:
                    continue

                if mod_path in module_cache:
                    mod_source, mod_tree = module_cache[mod_path]
                else:
                    mod_source = mod_path.read_text(encoding="utf-8")
                    mod_tree = ast.parse(mod_source)
                    module_cache[mod_path] = (mod_source, mod_tree)

                class_source = extract_full_class_from_module(type_name, mod_source, mod_tree)
                if class_source is None:
                    resolved_class = resolve_instance_class_name(type_name, mod_tree)
                    if resolved_class and resolved_class not in existing_class_names and resolved_class not in emitted:
                        class_source = extract_full_class_from_module(resolved_class, mod_source, mod_tree)
                        type_name = resolved_class
                if class_source:
                    code_strings.append(CodeString(code=class_source, file_path=mod_path))
                    emitted.add(type_name)
            except Exception:
                logger.debug(f"Error extracting type context class for {type_name} from {module_name}")
                continue

    return CodeStringsMarkdown(code_strings=code_strings)


def remove_docstring_from_body(indented_block: cst.IndentedBlock) -> cst.CSTNode:
    """Removes the docstring from an indented block if it exists."""
    if not isinstance(indented_block.body[0], cst.SimpleStatementLine):
        return indented_block
    first_stmt = indented_block.body[0].body[0]
    if isinstance(first_stmt, cst.Expr) and isinstance(first_stmt.value, cst.SimpleString):
        return indented_block.with_changes(body=indented_block.body[1:])
    return indented_block


def parse_code_and_prune_cst(
    code: str,
    code_context_type: CodeContextType,
    target_functions: set[str],
    helpers_of_helper_functions: set[str] = set(),  # noqa: B006
    remove_docstrings: bool = False,
) -> str:
    """Create a read-only version of the code by parsing and filtering the code to keep only class contextual information, and other module scoped variables."""
    module = cst.parse_module(code)
    defs_with_usages = collect_top_level_defs_with_usages(module, target_functions | helpers_of_helper_functions)

    if code_context_type == CodeContextType.READ_WRITABLE:
        filtered_node, found_target = prune_cst(
            module, target_functions, defs_with_usages=defs_with_usages, keep_class_init=True
        )
    elif code_context_type == CodeContextType.READ_ONLY:
        filtered_node, found_target = prune_cst(
            module,
            target_functions,
            helpers=helpers_of_helper_functions,
            remove_docstrings=remove_docstrings,
            include_target_in_output=False,
            include_dunder_methods=True,
        )
    elif code_context_type == CodeContextType.TESTGEN:
        filtered_node, found_target = prune_cst(
            module,
            target_functions,
            helpers=helpers_of_helper_functions,
            remove_docstrings=remove_docstrings,
            include_dunder_methods=True,
            include_init_dunder=True,
        )
    elif code_context_type == CodeContextType.HASHING:
        filtered_node, found_target = prune_cst(
            module, target_functions, remove_docstrings=True, exclude_init_from_targets=True
        )
    else:
        raise ValueError(f"Unknown code_context_type: {code_context_type}")  # noqa: EM102

    if not found_target:
        raise ValueError("No target functions found in the provided code")
    if filtered_node and isinstance(filtered_node, cst.Module):
        code = str(filtered_node.code)
        if code_context_type == CodeContextType.HASHING:
            code = ast.unparse(ast.parse(code))  # Makes it standard
        return code
    return ""


def prune_cst(
    node: cst.CSTNode,
    target_functions: set[str],
    prefix: str = "",
    *,
    defs_with_usages: dict[str, UsageInfo] | None = None,
    helpers: set[str] | None = None,
    remove_docstrings: bool = False,
    include_target_in_output: bool = True,
    exclude_init_from_targets: bool = False,
    keep_class_init: bool = False,
    include_dunder_methods: bool = False,
    include_init_dunder: bool = False,
) -> tuple[cst.CSTNode | None, bool]:
    """Unified function to prune CST nodes based on various filtering criteria.

    Args:
        node: The CST node to filter
        target_functions: Set of qualified function names that are targets
        prefix: Current qualified name prefix (for class methods)
        defs_with_usages: Dict of definitions with usage info (for READ_WRITABLE mode)
        helpers: Set of helper function qualified names (for READ_ONLY/TESTGEN modes)
        remove_docstrings: Whether to remove docstrings from output
        include_target_in_output: Whether to include target functions in output
        exclude_init_from_targets: Whether to exclude __init__ from targets (HASHING mode)
        keep_class_init: Whether to keep __init__ methods in classes (READ_WRITABLE mode)
        include_dunder_methods: Whether to include dunder methods (READ_ONLY/TESTGEN modes)
        include_init_dunder: Whether to include __init__ in dunder methods

    Returns:
        (filtered_node, found_target):
          filtered_node: The modified CST node or None if it should be removed.
          found_target: True if a target function was found in this node's subtree.

    """
    if isinstance(node, (cst.Import, cst.ImportFrom)):
        return None, False

    if isinstance(node, cst.FunctionDef):
        qualified_name = f"{prefix}.{node.name.value}" if prefix else node.name.value

        # Check if it's a helper function (higher priority than target)
        if helpers and qualified_name in helpers:
            if remove_docstrings and isinstance(node.body, cst.IndentedBlock):
                return node.with_changes(body=remove_docstring_from_body(node.body)), True
            return node, True

        # Check if it's a target function
        if qualified_name in target_functions:
            # Handle exclude_init_from_targets for HASHING mode
            if exclude_init_from_targets and node.name.value == "__init__":
                return None, False

            if include_target_in_output:
                if remove_docstrings and isinstance(node.body, cst.IndentedBlock):
                    return node.with_changes(body=remove_docstring_from_body(node.body)), True
                return node, True
            return None, True

        # Handle class __init__ for READ_WRITABLE mode
        if keep_class_init and node.name.value == "__init__":
            return node, False

        # Handle dunder methods for READ_ONLY/TESTGEN modes
        if (
            include_dunder_methods
            and len(node.name.value) > 4
            and node.name.value.startswith("__")
            and node.name.value.endswith("__")
        ):
            if not include_init_dunder and node.name.value == "__init__":
                return None, False
            if remove_docstrings and isinstance(node.body, cst.IndentedBlock):
                return node.with_changes(body=remove_docstring_from_body(node.body)), False
            return node, False

        return None, False

    if isinstance(node, cst.ClassDef):
        if prefix:
            return None, False
        if not isinstance(node.body, cst.IndentedBlock):
            raise ValueError("ClassDef body is not an IndentedBlock")  # noqa: TRY004
        class_prefix = node.name.value
        class_name = node.name.value

        # Handle dependency classes for READ_WRITABLE mode
        if defs_with_usages:
            # Check if this class contains any target functions
            has_target_functions = any(
                isinstance(stmt, cst.FunctionDef) and f"{class_prefix}.{stmt.name.value}" in target_functions
                for stmt in node.body.body
            )

            # If the class is used as a dependency (not containing target functions), keep it entirely
            if (
                not has_target_functions
                and class_name in defs_with_usages
                and defs_with_usages[class_name].used_by_qualified_function
            ):
                return node, True

        # Recursively filter each statement in the class body
        new_class_body: list[cst.CSTNode] = []
        found_in_class = False

        for stmt in node.body.body:
            filtered, found_target = prune_cst(
                stmt,
                target_functions,
                class_prefix,
                defs_with_usages=defs_with_usages,
                helpers=helpers,
                remove_docstrings=remove_docstrings,
                include_target_in_output=include_target_in_output,
                exclude_init_from_targets=exclude_init_from_targets,
                keep_class_init=keep_class_init,
                include_dunder_methods=include_dunder_methods,
                include_init_dunder=include_init_dunder,
            )
            found_in_class |= found_target
            if filtered:
                new_class_body.append(filtered)

        if not found_in_class:
            return None, False

        # Apply docstring removal to class if needed
        if remove_docstrings and new_class_body:
            updated_body = node.body.with_changes(body=new_class_body)
            assert isinstance(updated_body, cst.IndentedBlock)
            return node.with_changes(body=remove_docstring_from_body(updated_body)), True

        return node.with_changes(body=node.body.with_changes(body=new_class_body)) if new_class_body else None, True

    # Handle assignments for READ_WRITABLE mode
    if defs_with_usages is not None:
        if isinstance(node, (cst.Assign, cst.AnnAssign, cst.AugAssign)):
            if is_assignment_used(node, defs_with_usages):
                return node, True
            return None, False

    # For other nodes, recursively process children
    section_names = get_section_names(node)
    if not section_names:
        return node, False

    if helpers is not None:
        return recurse_sections(
            node,
            section_names,
            lambda child: prune_cst(
                child,
                target_functions,
                prefix,
                defs_with_usages=defs_with_usages,
                helpers=helpers,
                remove_docstrings=remove_docstrings,
                include_target_in_output=include_target_in_output,
                exclude_init_from_targets=exclude_init_from_targets,
                keep_class_init=keep_class_init,
                include_dunder_methods=include_dunder_methods,
                include_init_dunder=include_init_dunder,
            ),
            keep_non_target_children=True,
        )
    return recurse_sections(
        node,
        section_names,
        lambda child: prune_cst(
            child,
            target_functions,
            prefix,
            defs_with_usages=defs_with_usages,
            helpers=helpers,
            remove_docstrings=remove_docstrings,
            include_target_in_output=include_target_in_output,
            exclude_init_from_targets=exclude_init_from_targets,
            keep_class_init=keep_class_init,
            include_dunder_methods=include_dunder_methods,
            include_init_dunder=include_init_dunder,
        ),
    )
