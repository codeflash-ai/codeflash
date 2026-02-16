from __future__ import annotations

import ast
import hashlib
import os
from collections import defaultdict
from itertools import chain
from pathlib import Path
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
    extract_names_from_targets,
    get_section_names,
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
    from collections.abc import Callable

    from jedi.api.classes import Name

    from codeflash.languages.base import HelperFunction
    from codeflash.languages.python.context.unused_definition_remover import UsageInfo

# Error message constants
READ_WRITABLE_LIMIT_ERROR = "Read-writable code has exceeded token limit, cannot proceed"
TESTGEN_LIMIT_ERROR = "Testgen code context has exceeded token limit, cannot proceed"


def safe_relative_to(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return path


def build_testgen_context(
    helpers_of_fto_dict: dict[Path, set[FunctionSource]],
    helpers_of_helpers_dict: dict[Path, set[FunctionSource]],
    project_root_path: Path,
    *,
    remove_docstrings: bool = False,
    include_enrichment: bool = True,
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
    testgen_context = build_testgen_context(helpers_of_fto_dict, helpers_of_helpers_dict, project_root_path)

    if encoded_tokens_len(testgen_context.markdown) > testgen_token_limit:
        logger.debug("Testgen context exceeded token limit, removing docstrings")
        testgen_context = build_testgen_context(
            helpers_of_fto_dict, helpers_of_helpers_dict, project_root_path, remove_docstrings=True
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
    target_relative_path = safe_relative_to(function_to_optimize.file_path, project_root_path)

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

        helper_relative_path = safe_relative_to(file_path, project_root_path)

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
        return CodeString(code=code_context, file_path=safe_relative_to(file_path, project_root_path))
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
                        is_project_path(definition_path, project_root_path)
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

    collect_imports(tree.body)
    return tree, imported_names


def collect_existing_class_names(tree: ast.Module) -> set[str]:
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def enrich_testgen_context(code_context: CodeStringsMarkdown, project_root_path: Path) -> CodeStringsMarkdown:
    import jedi

    result = _parse_and_collect_imports(code_context)
    if result is None:
        return CodeStringsMarkdown(code_strings=[])
    tree, imported_names = result

    if not imported_names:
        return CodeStringsMarkdown(code_strings=[])

    existing_classes = collect_existing_class_names(tree)

    # Collect base class names from ClassDef nodes (single walk)
    base_class_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_class_names.add(base.id)
                elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
                    base_class_names.add(base.attr)

    # Classify external imports using importlib-based check
    is_project_cache: dict[str, bool] = {}
    external_base_classes: set[tuple[str, str]] = set()
    external_direct_imports: set[tuple[str, str]] = set()

    for name, module_name in imported_names.items():
        if not _is_project_module_cached(module_name, project_root_path, is_project_cache):
            if name in base_class_names:
                external_base_classes.add((name, module_name))
            if name not in existing_classes:
                external_direct_imports.add((name, module_name))

    code_strings: list[CodeString] = []
    emitted_class_names: set[str] = set()

    # --- Step 1: Project class definitions (jedi resolution + recursive base extraction) ---
    extracted_classes: set[tuple[Path, str]] = set()
    module_cache: dict[Path, tuple[str, ast.Module]] = {}

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

    def extract_class_and_bases(
        class_name: str, module_path: Path, module_source: str, module_tree: ast.Module
    ) -> None:
        if (module_path, class_name) in extracted_classes:
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

            if base_name and base_name not in existing_classes:
                extract_class_and_bases(base_name, module_path, module_source, module_tree)

        if (module_path, class_name) in extracted_classes:
            return

        lines = module_source.split("\n")
        start_line = class_node.lineno
        if class_node.decorator_list:
            start_line = min(d.lineno for d in class_node.decorator_list)
        class_source = "\n".join(lines[start_line - 1 : class_node.end_lineno])

        class_imports = extract_imports_for_class(module_tree, class_node, module_source)
        full_source = class_imports + "\n\n" + class_source if class_imports else class_source

        code_strings.append(CodeString(code=full_source, file_path=module_path))
        extracted_classes.add((module_path, class_name))
        emitted_class_names.add(class_name)

    for name, module_name in imported_names.items():
        if name in existing_classes:
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

            if not is_project_path(module_path, project_root_path):
                continue

            mod_result = get_module_source_and_tree(module_path)
            if mod_result is None:
                continue
            module_source, module_tree = mod_result

            extract_class_and_bases(name, module_path, module_source, module_tree)

        except Exception:
            logger.debug(f"Error extracting class definition for {name} from {module_name}")
            continue

    # --- Step 2: External base class __init__ stubs ---
    if external_base_classes:
        for cls, name in resolve_classes_from_modules(external_base_classes):
            if name in emitted_class_names:
                continue
            stub = extract_init_stub(cls, name, require_site_packages=False)
            if stub is not None:
                code_strings.append(stub)
                emitted_class_names.add(name)

    # --- Step 3: External direct import __init__ stubs with BFS ---
    if external_direct_imports:
        processed_classes: set[type] = set()
        worklist: list[tuple[type, str, int]] = [
            (cls, name, 0) for cls, name in resolve_classes_from_modules(external_direct_imports)
        ]

        while worklist:
            cls, class_name, depth = worklist.pop(0)

            if cls in processed_classes:
                continue
            processed_classes.add(cls)

            stub = extract_init_stub(cls, class_name)
            if stub is None:
                continue

            if class_name not in emitted_class_names:
                code_strings.append(stub)
                emitted_class_names.add(class_name)

            if depth < MAX_TRANSITIVE_DEPTH:
                for dep_cls in resolve_transitive_type_deps(cls):
                    if dep_cls not in processed_classes:
                        worklist.append((dep_cls, dep_cls.__name__, depth + 1))

    return CodeStringsMarkdown(code_strings=code_strings)


def resolve_classes_from_modules(candidates: set[tuple[str, str]]) -> list[tuple[type, str]]:
    """Import modules and resolve candidate (class_name, module_name) pairs to class objects."""
    import importlib
    import inspect

    resolved: list[tuple[type, str]] = []
    module_cache: dict[str, object] = {}

    for class_name, module_name in candidates:
        try:
            module = module_cache.get(module_name)
            if module is None:
                module = importlib.import_module(module_name)
                module_cache[module_name] = module

            cls = getattr(module, class_name, None)
            if cls is not None and inspect.isclass(cls):
                resolved.append((cls, class_name))
        except (ImportError, ModuleNotFoundError, AttributeError):
            logger.debug(f"Failed to import {module_name}.{class_name}")

    return resolved


MAX_TRANSITIVE_DEPTH = 5


def extract_classes_from_type_hint(hint: object) -> list[type]:
    """Recursively extract concrete class objects from a type annotation.

    Unwraps Optional, Union, List, Dict, Callable, Annotated, etc.
    Filters out builtins and typing module types.
    """
    import typing

    classes: list[type] = []
    origin = getattr(hint, "__origin__", None)
    args = getattr(hint, "__args__", None)

    if origin is not None and args:
        for arg in args:
            classes.extend(extract_classes_from_type_hint(arg))
    elif isinstance(hint, type):
        module = getattr(hint, "__module__", "")
        if module not in ("builtins", "typing", "typing_extensions", "types"):
            classes.append(hint)
    # Handle typing.Annotated on older Pythons where __origin__ may not be set
    if hasattr(typing, "get_args") and origin is None and args is None:
        try:
            inner_args = typing.get_args(hint)
            if inner_args:
                for arg in inner_args:
                    classes.extend(extract_classes_from_type_hint(arg))
        except Exception:
            pass

    return classes


def resolve_transitive_type_deps(cls: type) -> list[type]:
    """Find external classes referenced in cls.__init__ type annotations.

    Returns classes from site-packages that have a custom __init__.
    """
    import inspect
    import typing

    try:
        init_method = getattr(cls, "__init__")
        hints = typing.get_type_hints(init_method)
    except Exception:
        return []

    deps: list[type] = []
    for param_name, hint in hints.items():
        if param_name == "return":
            continue
        for dep_cls in extract_classes_from_type_hint(hint):
            if dep_cls is cls:
                continue
            init_method = getattr(dep_cls, "__init__", None)
            if init_method is None or init_method is object.__init__:
                continue
            try:
                class_file = Path(inspect.getfile(dep_cls))
            except (OSError, TypeError):
                continue
            if not path_belongs_to_site_packages(class_file):
                continue
            deps.append(dep_cls)

    return deps


def extract_init_stub(cls: type, class_name: str, require_site_packages: bool = True) -> CodeString | None:
    """Extract a stub containing the class definition with only its __init__ method.

    Args:
        cls: The class object to extract __init__ from
        class_name: Name to use for the class in the stub
        require_site_packages: If True, only extract from site-packages. If False, include stdlib too.

    """
    import inspect
    import textwrap

    init_method = getattr(cls, "__init__", None)
    if init_method is None or init_method is object.__init__:
        return None

    try:
        class_file = Path(inspect.getfile(cls))
    except (OSError, TypeError):
        return None

    if require_site_packages and not path_belongs_to_site_packages(class_file):
        return None

    try:
        init_source = inspect.getsource(init_method)
        init_source = textwrap.dedent(init_source)
    except (OSError, TypeError):
        return None

    parts = class_file.parts
    if "site-packages" in parts:
        idx = parts.index("site-packages")
        class_file = Path(*parts[idx + 1 :])

    class_source = f"class {class_name}:\n" + textwrap.indent(init_source, "    ")
    return CodeString(code=class_source, file_path=class_file)


def _is_project_module_cached(module_name: str, project_root_path: Path, cache: dict[str, bool]) -> bool:
    cached = cache.get(module_name)
    if cached is not None:
        return cached
    is_project = _is_project_module(module_name, project_root_path)
    cache[module_name] = is_project
    return is_project


def is_project_path(module_path: Path | None, project_root_path: Path) -> bool:
    if module_path is None:
        return False
    # site-packages must be checked first because .venv/site-packages is under project root
    if path_belongs_to_site_packages(module_path):
        return False
    return str(module_path).startswith(str(project_root_path) + os.sep)


def _is_project_module(module_name: str, project_root_path: Path) -> bool:
    """Check if a module is part of the project (not external/stdlib)."""
    import importlib.util

    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return False
    else:
        if spec is None or spec.origin is None:
            return False
        return is_project_path(Path(spec.origin), project_root_path)


def extract_imports_for_class(module_tree: ast.Module, class_node: ast.ClassDef, module_source: str) -> str:
    """Extract import statements needed for a class definition.

    This extracts imports for base classes, decorators, and type annotations.
    """
    needed_names: set[str] = set()

    # Get base class names
    for base in class_node.bases:
        if isinstance(base, ast.Name):
            needed_names.add(base.id)
        elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
            # For things like abc.ABC, we need the module name
            needed_names.add(base.value.id)

    # Get decorator names (e.g., dataclass, field)
    for decorator in class_node.decorator_list:
        if isinstance(decorator, ast.Name):
            needed_names.add(decorator.id)
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                needed_names.add(decorator.func.id)
            elif isinstance(decorator.func, ast.Attribute) and isinstance(decorator.func.value, ast.Name):
                needed_names.add(decorator.func.value.id)

    # Get type annotation names from class body (for dataclass fields)
    for item in class_node.body:
        if isinstance(item, ast.AnnAssign) and item.annotation:
            collect_names_from_annotation(item.annotation, needed_names)
        # Also check for field() calls which are common in dataclasses
        elif isinstance(item, ast.Assign) and isinstance(item.value, ast.Call):
            if isinstance(item.value.func, ast.Name):
                needed_names.add(item.value.func.id)

    # Find imports that provide these names
    import_lines: list[str] = []
    source_lines = module_source.split("\n")
    added_imports: set[int] = set()  # Track line numbers to avoid duplicates

    for node in module_tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name.split(".")[0]
                if name in needed_names and node.lineno not in added_imports:
                    import_lines.append(source_lines[node.lineno - 1])
                    added_imports.add(node.lineno)
                    break
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                if name in needed_names and node.lineno not in added_imports:
                    import_lines.append(source_lines[node.lineno - 1])
                    added_imports.add(node.lineno)
                    break

    return "\n".join(import_lines)


def collect_names_from_annotation(node: ast.expr, names: set[str]) -> None:
    """Recursively collect type annotation names from an AST node."""
    if isinstance(node, ast.Name):
        names.add(node.id)
    elif isinstance(node, ast.Subscript):
        collect_names_from_annotation(node.value, names)
        collect_names_from_annotation(node.slice, names)
    elif isinstance(node, ast.Tuple):
        for elt in node.elts:
            collect_names_from_annotation(elt, names)
    elif isinstance(node, ast.BinOp):  # For Union types with | syntax
        collect_names_from_annotation(node.left, names)
        collect_names_from_annotation(node.right, names)
    elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        names.add(node.value.id)


def is_dunder_method(name: str) -> bool:
    return len(name) > 4 and name.isascii() and name.startswith("__") and name.endswith("__")


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


def _qualified_name(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _validate_classdef(node: cst.ClassDef, prefix: str) -> tuple[str, cst.IndentedBlock] | None:
    if prefix:
        return None
    if not isinstance(node.body, cst.IndentedBlock):
        raise ValueError("ClassDef body is not an IndentedBlock")  # noqa: TRY004
    return _qualified_name(prefix, node.name.value), node.body


def _recurse_sections(
    node: cst.CSTNode,
    section_names: list[str],
    prune_fn: Callable[[cst.CSTNode], tuple[cst.CSTNode | None, bool]],
    keep_non_target_children: bool = False,
) -> tuple[cst.CSTNode | None, bool]:
    updates: dict[str, list[cst.CSTNode] | cst.CSTNode] = {}
    found_any_target = False
    for section in section_names:
        original_content = getattr(node, section, None)
        if isinstance(original_content, (list, tuple)):
            new_children = []
            section_found_target = False
            for child in original_content:
                filtered, found_target = prune_fn(child)
                if filtered:
                    new_children.append(filtered)
                section_found_target |= found_target
            if keep_non_target_children:
                if section_found_target or new_children:
                    found_any_target |= section_found_target
                    updates[section] = new_children
            elif section_found_target:
                found_any_target = True
                updates[section] = new_children
        elif original_content is not None:
            filtered, found_target = prune_fn(original_content)
            if keep_non_target_children:
                found_any_target |= found_target
                if filtered:
                    updates[section] = filtered
            elif found_target:
                found_any_target = True
                if filtered:
                    updates[section] = filtered
    if keep_non_target_children:
        if updates:
            return node.with_changes(**updates), found_any_target
        return None, False
    if not found_any_target:
        return None, False
    return (node.with_changes(**updates) if updates else node), True


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
        qualified_name = _qualified_name(prefix, node.name.value)

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
        if include_dunder_methods and is_dunder_method(node.name.value):
            if not include_init_dunder and node.name.value == "__init__":
                return None, False
            if remove_docstrings and isinstance(node.body, cst.IndentedBlock):
                return node.with_changes(body=remove_docstring_from_body(node.body)), False
            return node, False

        return None, False

    if isinstance(node, cst.ClassDef):
        result = _validate_classdef(node, prefix)
        if result is None:
            return None, False
        class_prefix, _ = result
        class_name = node.name.value

        # Handle dependency classes for READ_WRITABLE mode
        if defs_with_usages:
            # Check if this class contains any target functions
            has_target_functions = any(
                isinstance(stmt, cst.FunctionDef) and _qualified_name(class_prefix, stmt.name.value) in target_functions
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
        if isinstance(node, cst.Assign):
            for target in node.targets:
                names = extract_names_from_targets(target.target)
                for name in names:
                    if name in defs_with_usages and defs_with_usages[name].used_by_qualified_function:
                        return node, True
            return None, False

        if isinstance(node, (cst.AnnAssign, cst.AugAssign)):
            names = extract_names_from_targets(node.target)
            for name in names:
                if name in defs_with_usages and defs_with_usages[name].used_by_qualified_function:
                    return node, True
            return None, False

    # For other nodes, recursively process children
    section_names = get_section_names(node)
    if not section_names:
        return node, False

    if helpers is not None:
        return _recurse_sections(
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
    return _recurse_sections(
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
