from __future__ import annotations

import ast
import hashlib
import os
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, cast

import libcst as cst

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_extractor import add_needed_imports_from_module, find_preexisting_objects
from codeflash.code_utils.code_utils import encoded_tokens_len, get_qualified_name, path_belongs_to_site_packages
from codeflash.code_utils.config_consts import OPTIMIZATION_CONTEXT_TOKEN_LIMIT, TESTGEN_CONTEXT_TOKEN_LIMIT
from codeflash.context.unused_definition_remover import (
    collect_top_level_defs_with_usages,
    extract_names_from_targets,
    get_section_names,
    remove_unused_definitions_by_function_names,
)
from codeflash.discovery.functions_to_optimize import FunctionToOptimize  # noqa: TC001

# Language support imports for multi-language code context extraction
from codeflash.languages import Language, is_python
from codeflash.models.models import (
    CodeContextType,
    CodeOptimizationContext,
    CodeString,
    CodeStringsMarkdown,
    FunctionSource,
)
from codeflash.optimization.function_context import belongs_to_function_qualified

if TYPE_CHECKING:
    from jedi.api.classes import Name
    from libcst import CSTNode

    from codeflash.context.unused_definition_remover import UsageInfo
    from codeflash.languages.base import HelperFunction


def build_testgen_context(
    helpers_of_fto_dict: dict[Path, set[FunctionSource]],
    helpers_of_helpers_dict: dict[Path, set[FunctionSource]],
    project_root_path: Path,
    remove_docstrings: bool,
    include_imported_classes: bool,
) -> CodeStringsMarkdown:
    """Build testgen context with optional imported class definitions and external base inits."""
    testgen_context = extract_code_markdown_context_from_files(
        helpers_of_fto_dict,
        helpers_of_helpers_dict,
        project_root_path,
        remove_docstrings=remove_docstrings,
        code_context_type=CodeContextType.TESTGEN,
    )

    if include_imported_classes:
        imported_class_context = get_imported_class_definitions(testgen_context, project_root_path)
        if imported_class_context.code_strings:
            testgen_context = CodeStringsMarkdown(
                code_strings=testgen_context.code_strings + imported_class_context.code_strings
            )

        external_base_inits = get_external_base_class_inits(testgen_context, project_root_path)
        if external_base_inits.code_strings:
            testgen_context = CodeStringsMarkdown(
                code_strings=testgen_context.code_strings + external_base_inits.code_strings
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
        raise ValueError("Read-writable code has exceeded token limit, cannot proceed")

    # Setup preexisting objects for code replacer
    preexisting_objects = set(
        chain(
            *(find_preexisting_objects(codestring.code) for codestring in final_read_writable_code.code_strings),
            *(find_preexisting_objects(codestring.code) for codestring in read_only_code_markdown.code_strings),
        )
    )
    read_only_context_code = read_only_code_markdown.markdown

    read_only_code_markdown_tokens = encoded_tokens_len(read_only_context_code)
    total_tokens = final_read_writable_tokens + read_only_code_markdown_tokens
    if total_tokens > optim_token_limit:
        logger.debug("Code context has exceeded token limit, removing docstrings from read-only code")
        # Extract read only code without docstrings
        read_only_code_no_docstring_markdown = extract_code_markdown_context_from_files(
            helpers_of_fto_dict, helpers_of_helpers_dict, project_root_path, remove_docstrings=True
        )
        read_only_context_code = read_only_code_no_docstring_markdown.markdown
        read_only_code_no_docstring_markdown_tokens = encoded_tokens_len(read_only_context_code)
        total_tokens = final_read_writable_tokens + read_only_code_no_docstring_markdown_tokens
        if total_tokens > optim_token_limit:
            logger.debug("Code context has exceeded token limit, removing read-only code")
            read_only_context_code = ""

    # Extract code context for testgen with progressive fallback for token limits
    # Try in order: full context -> remove docstrings -> remove imported classes
    testgen_context = build_testgen_context(
        helpers_of_fto_dict,
        helpers_of_helpers_dict,
        project_root_path,
        remove_docstrings=False,
        include_imported_classes=True,
    )

    if encoded_tokens_len(testgen_context.markdown) > testgen_token_limit:
        logger.debug("Testgen context exceeded token limit, removing docstrings")
        testgen_context = build_testgen_context(
            helpers_of_fto_dict,
            helpers_of_helpers_dict,
            project_root_path,
            remove_docstrings=True,
            include_imported_classes=True,
        )

        if encoded_tokens_len(testgen_context.markdown) > testgen_token_limit:
            logger.debug("Testgen context still exceeded token limit, removing imported class definitions")
            testgen_context = build_testgen_context(
                helpers_of_fto_dict,
                helpers_of_helpers_dict,
                project_root_path,
                remove_docstrings=True,
                include_imported_classes=False,
            )

            if encoded_tokens_len(testgen_context.markdown) > testgen_token_limit:
                raise ValueError("Testgen code context has exceeded token limit, cannot proceed")
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
        raise ValueError("Read-writable code has exceeded token limit, cannot proceed")

    testgen_tokens = encoded_tokens_len(testgen_context.markdown)
    if testgen_tokens > testgen_token_limit:
        raise ValueError("Testgen code context has exceeded token limit, cannot proceed")

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
        try:
            original_code = file_path.read_text("utf8")
        except Exception as e:
            logger.exception(f"Error while parsing {file_path}: {e}")
            continue
        try:
            qualified_function_names = {func.qualified_name for func in function_sources}
            helpers_of_helpers_qualified_names = {
                func.qualified_name for func in helpers_of_helpers.get(file_path, set())
            }
            code_without_unused_defs = remove_unused_definitions_by_function_names(
                original_code, qualified_function_names | helpers_of_helpers_qualified_names
            )
            code_context = parse_code_and_prune_cst(
                code_without_unused_defs,
                code_context_type,
                qualified_function_names,
                helpers_of_helpers_qualified_names,
                remove_docstrings,
            )

        except ValueError as e:
            logger.debug(f"Error while getting read-only code: {e}")
            continue
        if code_context.strip():
            if code_context_type != CodeContextType.HASHING:
                code_context = add_needed_imports_from_module(
                    src_module_code=original_code,
                    dst_module_code=code_context,
                    src_path=file_path,
                    dst_path=file_path,
                    project_root=project_root_path,
                    helper_functions=list(
                        helpers_of_fto.get(file_path, set()) | helpers_of_helpers.get(file_path, set())
                    ),
                )
            code_string_context = CodeString(
                code=code_context, file_path=file_path.resolve().relative_to(project_root_path.resolve())
            )
            code_context_markdown.code_strings.append(code_string_context)
    # Extract code from file paths containing helpers of helpers
    for file_path, helper_function_sources in helpers_of_helpers_no_overlap.items():
        try:
            original_code = file_path.read_text("utf8")
        except Exception as e:
            logger.exception(f"Error while parsing {file_path}: {e}")
            continue
        try:
            qualified_helper_function_names = {func.qualified_name for func in helper_function_sources}
            code_without_unused_defs = remove_unused_definitions_by_function_names(
                original_code, qualified_helper_function_names
            )
            code_context = parse_code_and_prune_cst(
                code_without_unused_defs, code_context_type, set(), qualified_helper_function_names, remove_docstrings
            )
        except ValueError as e:
            logger.debug(f"Error while getting read-only code: {e}")
            continue

        if code_context.strip():
            if code_context_type != CodeContextType.HASHING:
                code_context = add_needed_imports_from_module(
                    src_module_code=original_code,
                    dst_module_code=code_context,
                    src_path=file_path,
                    dst_path=file_path,
                    project_root=project_root_path,
                    helper_functions=list(helpers_of_helpers_no_overlap.get(file_path, set())),
                )
            code_string_context = CodeString(
                code=code_context, file_path=file_path.resolve().relative_to(project_root_path.resolve())
            )
            code_context_markdown.code_strings.append(code_string_context)
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
                        str(definition_path).startswith(str(project_root_path) + os.sep)
                        and not path_belongs_to_site_packages(definition_path)
                        and definition.full_name
                        and not belongs_to_function_qualified(definition, qualified_function_name)
                        and definition.full_name.startswith(definition.module_name)
                    )
                    if is_valid_definition and definition.type == "function":
                        qualified_name = get_qualified_name(definition.module_name, definition.full_name)
                        # Avoid nested functions or classes. Only class.function is allowed
                        if len(qualified_name.split(".")) <= 2:
                            function_source = FunctionSource(
                                file_path=definition_path,
                                qualified_name=qualified_name,
                                fully_qualified_name=definition.full_name,
                                only_function_name=definition.name,
                                source_code=definition.get_line_code(),
                                jedi_definition=definition,
                            )
                            file_path_to_function_source[definition_path].add(function_source)
                            function_source_list.append(function_source)
                    # When a class is instantiated (e.g., MyClass()), track its __init__ as a helper
                    # This ensures the class definition with constructor is included in testgen context
                    elif is_valid_definition and definition.type == "class":
                        init_qualified_name = get_qualified_name(
                            definition.module_name, f"{definition.full_name}.__init__"
                        )
                        # Only include if it's a top-level class (not nested)
                        if len(init_qualified_name.split(".")) <= 2:
                            function_source = FunctionSource(
                                file_path=definition_path,
                                qualified_name=init_qualified_name,
                                fully_qualified_name=f"{definition.full_name}.__init__",
                                only_function_name="__init__",
                                source_code=definition.get_line_code(),
                                jedi_definition=definition,
                            )
                            file_path_to_function_source[definition_path].add(function_source)
                            function_source_list.append(function_source)

    return file_path_to_function_source, function_source_list


def get_imported_class_definitions(code_context: CodeStringsMarkdown, project_root_path: Path) -> CodeStringsMarkdown:
    """Extract class definitions for imported types from project modules.

    This function analyzes the imports in the extracted code context and fetches
    class definitions for any classes imported from project modules. This helps
    the LLM understand the actual class structure (constructors, methods, inheritance)
    rather than just seeing import statements.

    Also recursively extracts base classes when a class inherits from another class
    in the same module, ensuring the full inheritance chain is available for
    understanding constructor signatures.

    Args:
        code_context: The already extracted code context containing imports
        project_root_path: Root path of the project

    Returns:
        CodeStringsMarkdown containing class definitions from imported project modules

    """
    import jedi

    # Collect all code from the context
    all_code = "\n".join(cs.code for cs in code_context.code_strings)

    # Parse to find import statements
    try:
        tree = ast.parse(all_code)
    except SyntaxError:
        return CodeStringsMarkdown(code_strings=[])

    # Collect imported names and their source modules
    imported_names: dict[str, str] = {}  # name -> module_path
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name != "*":
                    imported_name = alias.asname or alias.name
                    imported_names[imported_name] = node.module

    if not imported_names:
        return CodeStringsMarkdown(code_strings=[])

    # Track which classes we've already extracted to avoid duplicates
    extracted_classes: set[tuple[Path, str]] = set()  # (file_path, class_name)

    # Also track what's already defined in the context
    existing_definitions: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            existing_definitions.add(node.name)

    class_code_strings: list[CodeString] = []

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
        """Extract a class and its base classes recursively from the same module."""
        # Skip if already extracted
        if (module_path, class_name) in extracted_classes:
            return

        # Find the class definition in the module
        class_node = None
        for node in ast.walk(module_tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                class_node = node
                break

        if class_node is None:
            return

        # First, recursively extract base classes from the same module
        for base in class_node.bases:
            base_name = None
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                # For module.ClassName, we skip (cross-module inheritance)
                continue

            if base_name and base_name not in existing_definitions:
                # Check if base class is defined in the same module
                extract_class_and_bases(base_name, module_path, module_source, module_tree)

        # Now extract this class (after its bases, so base classes appear first)
        if (module_path, class_name) in extracted_classes:
            return  # Already added by another path

        lines = module_source.split("\n")
        start_line = class_node.lineno
        if class_node.decorator_list:
            start_line = min(d.lineno for d in class_node.decorator_list)
        class_source = "\n".join(lines[start_line - 1 : class_node.end_lineno])

        # Extract imports for the class
        class_imports = extract_imports_for_class(module_tree, class_node, module_source)
        full_source = class_imports + "\n\n" + class_source if class_imports else class_source

        class_code_strings.append(CodeString(code=full_source, file_path=module_path))
        extracted_classes.add((module_path, class_name))

    for name, module_name in imported_names.items():
        # Skip if already defined in context
        if name in existing_definitions:
            continue

        # Try to find the module file using Jedi
        try:
            # Create a script that imports the module to resolve it
            test_code = f"import {module_name}"
            script = jedi.Script(test_code, project=jedi.Project(path=project_root_path))
            completions = script.goto(1, len(test_code))

            if not completions:
                continue

            module_path = completions[0].module_path
            if not module_path:
                continue

            # Check if this is a project module (not stdlib/third-party)
            if not str(module_path).startswith(str(project_root_path) + os.sep):
                continue
            if path_belongs_to_site_packages(module_path):
                continue

            # Get module source and tree
            result = get_module_source_and_tree(module_path)
            if result is None:
                continue
            module_source, module_tree = result

            # Extract the class and its base classes
            extract_class_and_bases(name, module_path, module_source, module_tree)

        except Exception:
            logger.debug(f"Error extracting class definition for {name} from {module_name}")
            continue

    return CodeStringsMarkdown(code_strings=class_code_strings)


def get_external_base_class_inits(code_context: CodeStringsMarkdown, project_root_path: Path) -> CodeStringsMarkdown:
    """Extract __init__ methods from external library base classes.

    Scans the code context for classes that inherit from external libraries and extracts
    just their __init__ methods. This helps the LLM understand constructor signatures
    for mocking or instantiation.
    """
    import importlib
    import inspect
    import textwrap

    all_code = "\n".join(cs.code for cs in code_context.code_strings)

    try:
        tree = ast.parse(all_code)
    except SyntaxError:
        return CodeStringsMarkdown(code_strings=[])

    imported_names: dict[str, str] = {}
    # Use a set to deduplicate external base entries to avoid repeated expensive checks/imports.
    external_bases_set: set[tuple[str, str]] = set()
    # Local cache to avoid repeated _is_project_module calls for the same module_name.
    is_project_cache: dict[str, bool] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name != "*":
                    imported_name = alias.asname or alias.name
                    imported_names[imported_name] = node.module
        elif isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = None
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
                    base_name = base.attr

                if base_name and base_name in imported_names:
                    module_name = imported_names[base_name]
                    # Check cache first to avoid repeated expensive checks.
                    cached = is_project_cache.get(module_name)
                    if cached is None:
                        is_project = _is_project_module(module_name, project_root_path)
                        is_project_cache[module_name] = is_project
                    else:
                        is_project = cached

                    if not is_project:
                        external_bases_set.add((base_name, module_name))

    if not external_bases_set:
        return CodeStringsMarkdown(code_strings=[])

    code_strings: list[CodeString] = []
    # Cache imported modules to avoid repeated importlib.import_module calls.
    imported_module_cache: dict[str, object] = {}

    for base_name, module_name in external_bases_set:

        try:
            module = imported_module_cache.get(module_name)
            if module is None:
                module = importlib.import_module(module_name)
                imported_module_cache[module_name] = module

            base_class = getattr(module, base_name, None)
            if base_class is None:
                continue

            init_method = getattr(base_class, "__init__", None)
            if init_method is None:
                continue

            try:
                init_source = inspect.getsource(init_method)
                init_source = textwrap.dedent(init_source)
                class_file = Path(inspect.getfile(base_class))
                parts = class_file.parts
                if "site-packages" in parts:
                    idx = parts.index("site-packages")
                    class_file = Path(*parts[idx + 1 :])
            except (OSError, TypeError):
                continue

            class_source = f"class {base_name}:\n" + textwrap.indent(init_source, "    ")
            code_strings.append(CodeString(code=class_source, file_path=class_file))

        except (ImportError, ModuleNotFoundError, AttributeError):
            logger.debug(f"Failed to extract __init__ for {module_name}.{base_name}")
            continue

    return CodeStringsMarkdown(code_strings=code_strings)


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
        module_path = Path(spec.origin)
        # Check if the module is in site-packages (external dependency)
        # This must be checked first because .venv/site-packages is under project root
        if path_belongs_to_site_packages(module_path):
            return False
        # Check if the module is within the project root
        return str(module_path).startswith(str(project_root_path) + os.sep)


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
    for item in ast.walk(class_node):
        if isinstance(item, ast.AnnAssign) and item.annotation:
            collect_names_from_annotation(item.annotation, needed_names)
        # Also check for field() calls which are common in dataclasses
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Name):
            needed_names.add(item.func.id)

    # Find imports that provide these names
    import_lines: list[str] = []
    source_lines = module_source.split("\n")
    added_imports: set[int] = set()  # Track line numbers to avoid duplicates

    for node in module_tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                if name in needed_names and node.lineno not in added_imports:
                    import_lines.append(source_lines[node.lineno - 1])
                    added_imports.add(node.lineno)
                    break
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname or alias.name
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


class UsedNameCollector(cst.CSTVisitor):
    """Collects all base names referenced in code (for import preservation)."""

    def __init__(self) -> None:
        self.used_names: set[str] = set()
        self.defined_names: set[str] = set()

    def visit_Name(self, node: cst.Name) -> None:
        self.used_names.add(node.value)

    def visit_Attribute(self, node: cst.Attribute) -> bool | None:
        base = node.value
        while isinstance(base, cst.Attribute):
            base = base.value
        if isinstance(base, cst.Name):
            self.used_names.add(base.value)
        return True

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        self.defined_names.add(node.name.value)
        return True

    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        self.defined_names.add(node.name.value)
        return True

    def visit_Assign(self, node: cst.Assign) -> bool | None:
        for target in node.targets:
            names = extract_names_from_targets(target.target)
            self.defined_names.update(names)
        return True

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool | None:
        names = extract_names_from_targets(node.target)
        self.defined_names.update(names)
        return True

    def get_external_names(self) -> set[str]:
        return self.used_names - self.defined_names - {"self", "cls"}


def get_imported_names(import_node: cst.Import | cst.ImportFrom) -> set[str]:
    """Extract the names made available by an import statement."""
    names: set[str] = set()
    if isinstance(import_node, cst.Import):
        if isinstance(import_node.names, cst.ImportStar):
            return {"*"}
        for alias in import_node.names:
            if isinstance(alias, cst.ImportAlias):
                if alias.asname and isinstance(alias.asname.name, cst.Name):
                    names.add(alias.asname.name.value)
                elif isinstance(alias.name, cst.Name):
                    names.add(alias.name.value)
                elif isinstance(alias.name, cst.Attribute):
                    # import foo.bar -> accessible as "foo"
                    base: cst.BaseExpression = alias.name
                    while isinstance(base, cst.Attribute):
                        base = base.value
                    if isinstance(base, cst.Name):
                        names.add(base.value)
    elif isinstance(import_node, cst.ImportFrom):
        if isinstance(import_node.names, cst.ImportStar):
            return {"*"}
        for alias in import_node.names:
            if isinstance(alias, cst.ImportAlias):
                if alias.asname and isinstance(alias.asname.name, cst.Name):
                    names.add(alias.asname.name.value)
                elif isinstance(alias.name, cst.Name):
                    names.add(alias.name.value)
    return names


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
        filtered_node, found_target = prune_cst_for_read_writable_code(module, target_functions, defs_with_usages)
    elif code_context_type == CodeContextType.READ_ONLY:
        filtered_node, found_target = prune_cst_for_context(
            module,
            target_functions,
            helpers_of_helper_functions,
            remove_docstrings=remove_docstrings,
            include_target_in_output=False,
            include_init_dunder=False,
        )
    elif code_context_type == CodeContextType.TESTGEN:
        filtered_node, found_target = prune_cst_for_context(
            module,
            target_functions,
            helpers_of_helper_functions,
            remove_docstrings=remove_docstrings,
            include_target_in_output=True,
            include_init_dunder=True,
        )
    elif code_context_type == CodeContextType.HASHING:
        filtered_node, found_target = prune_cst_for_code_hashing(module, target_functions)
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


def prune_cst_for_read_writable_code(
    node: cst.CSTNode, target_functions: set[str], defs_with_usages: dict[str, UsageInfo], prefix: str = ""
) -> tuple[cst.CSTNode | None, bool]:
    """Recursively filter the node and its children to build the read-writable codeblock. This contains nodes that lead to target functions.

    Returns
    -------
        (filtered_node, found_target):
          filtered_node: The modified CST node or None if it should be removed.
          found_target: True if a target function was found in this node's subtree.

    """
    if isinstance(node, (cst.Import, cst.ImportFrom)):
        return None, False

    if isinstance(node, cst.FunctionDef):
        qualified_name = f"{prefix}.{node.name.value}" if prefix else node.name.value
        if qualified_name in target_functions:
            return node, True
        return None, False

    if isinstance(node, cst.ClassDef):
        # Do not recurse into nested classes
        if prefix:
            return None, False

        class_name = node.name.value

        # Assuming always an IndentedBlock
        if not isinstance(node.body, cst.IndentedBlock):
            raise ValueError("ClassDef body is not an IndentedBlock")  # noqa: TRY004
        class_prefix = f"{prefix}.{class_name}" if prefix else class_name

        # Check if this class contains any target functions
        has_target_functions = any(
            isinstance(stmt, cst.FunctionDef) and f"{class_prefix}.{stmt.name.value}" in target_functions
            for stmt in node.body.body
        )

        # If the class is used as a dependency (not containing target functions), keep it entirely
        # This handles cases like enums, dataclasses, and other types used by the target function
        if (
            not has_target_functions
            and class_name in defs_with_usages
            and defs_with_usages[class_name].used_by_qualified_function
        ):
            return node, True

        new_body = []
        found_target = False

        for stmt in node.body.body:
            if isinstance(stmt, cst.FunctionDef):
                qualified_name = f"{class_prefix}.{stmt.name.value}"
                if qualified_name in target_functions:
                    new_body.append(stmt)
                    found_target = True
                elif stmt.name.value == "__init__":
                    new_body.append(stmt)  # enable __init__ optimizations
        # If no target functions found, remove the class entirely
        if not new_body or not found_target:
            return None, False

        return node.with_changes(body=cst.IndentedBlock(body=new_body)), found_target

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

    # For other nodes, we preserve them only if they contain target functions in their children.
    section_names = get_section_names(node)
    if not section_names:
        return node, False

    updates: dict[str, list[cst.CSTNode] | cst.CSTNode] = {}
    found_any_target = False

    for section in section_names:
        original_content = getattr(node, section, None)
        if isinstance(original_content, (list, tuple)):
            new_children = []
            section_found_target = False
            for child in original_content:
                filtered, found_target = prune_cst_for_read_writable_code(
                    child, target_functions, defs_with_usages, prefix
                )
                if filtered:
                    new_children.append(filtered)
                section_found_target |= found_target

            if section_found_target:
                found_any_target = True
                updates[section] = new_children
        elif original_content is not None:
            filtered, found_target = prune_cst_for_read_writable_code(
                original_content, target_functions, defs_with_usages, prefix
            )
            if found_target:
                found_any_target = True
                if filtered:
                    updates[section] = filtered

    if not found_any_target:
        return None, False
    return (node.with_changes(**updates) if updates else node), True


def prune_cst_for_code_hashing(
    node: cst.CSTNode, target_functions: set[str], prefix: str = ""
) -> tuple[cst.CSTNode | None, bool]:
    """Recursively filter the node and its children to build the read-writable codeblock. This contains nodes that lead to target functions.

    Returns
    -------
        (filtered_node, found_target):
          filtered_node: The modified CST node or None if it should be removed.
          found_target: True if a target function was found in this node's subtree.

    """
    if isinstance(node, (cst.Import, cst.ImportFrom)):
        return None, False

    if isinstance(node, cst.FunctionDef):
        qualified_name = f"{prefix}.{node.name.value}" if prefix else node.name.value
        # For hashing, exclude __init__ methods even if in target_functions
        # because they don't affect the semantic behavior being hashed
        # But include other dunder methods like __call__ which do affect behavior
        if qualified_name in target_functions and node.name.value != "__init__":
            new_body = remove_docstring_from_body(node.body) if isinstance(node.body, cst.IndentedBlock) else node.body
            return node.with_changes(body=new_body), True
        return None, False

    if isinstance(node, cst.ClassDef):
        # Do not recurse into nested classes
        if prefix:
            return None, False
        # Assuming always an IndentedBlock
        if not isinstance(node.body, cst.IndentedBlock):
            raise ValueError("ClassDef body is not an IndentedBlock")  # noqa: TRY004
        class_prefix = f"{prefix}.{node.name.value}" if prefix else node.name.value
        new_class_body: list[cst.CSTNode] = []
        found_target = False

        for stmt in node.body.body:
            if isinstance(stmt, cst.FunctionDef):
                qualified_name = f"{class_prefix}.{stmt.name.value}"
                # For hashing, exclude __init__ methods even if in target_functions
                # but include other methods like __call__ which affect behavior
                if qualified_name in target_functions and stmt.name.value != "__init__":
                    stmt_with_changes = stmt.with_changes(
                        body=remove_docstring_from_body(cast("cst.IndentedBlock", stmt.body))
                    )
                    new_class_body.append(stmt_with_changes)
                    found_target = True
        # If no target functions found, remove the class entirely
        if not new_class_body or not found_target:
            return None, False
        return node.with_changes(
            body=cst.IndentedBlock(cast("list[cst.BaseStatement]", new_class_body))
        ) if new_class_body else None, found_target

    # For other nodes, we preserve them only if they contain target functions in their children.
    section_names = get_section_names(node)
    if not section_names:
        return node, False

    updates: dict[str, list[cst.CSTNode] | cst.CSTNode] = {}
    found_any_target = False

    for section in section_names:
        original_content = getattr(node, section, None)
        if isinstance(original_content, (list, tuple)):
            new_children = []
            section_found_target = False
            for child in original_content:
                filtered, found_target = prune_cst_for_code_hashing(child, target_functions, prefix)
                if filtered:
                    new_children.append(filtered)
                section_found_target |= found_target

            if section_found_target:
                found_any_target = True
                updates[section] = new_children
        elif original_content is not None:
            filtered, found_target = prune_cst_for_code_hashing(original_content, target_functions, prefix)
            if found_target:
                found_any_target = True
                if filtered:
                    updates[section] = filtered

    if not found_any_target:
        return None, False

    return (node.with_changes(**updates) if updates else node), True


def prune_cst_for_context(
    node: cst.CSTNode,
    target_functions: set[str],
    helpers_of_helper_functions: set[str],
    prefix: str = "",
    remove_docstrings: bool = False,
    include_target_in_output: bool = False,
    include_init_dunder: bool = False,
) -> tuple[cst.CSTNode | None, bool]:
    """Recursively filter the node for code context extraction.

    Args:
        node: The CST node to filter
        target_functions: Set of qualified function names that are targets
        helpers_of_helper_functions: Set of helper function qualified names
        prefix: Current qualified name prefix (for class methods)
        remove_docstrings: Whether to remove docstrings from output
        include_target_in_output: If True, include target functions in output (testgen mode)
                                  If False, exclude target functions (read-only mode)
        include_init_dunder: If True, include __init__ in dunder methods (testgen mode)
                             If False, exclude __init__ from dunder methods (read-only mode)

    Returns:
        (filtered_node, found_target):
          filtered_node: The modified CST node or None if it should be removed.
          found_target: True if a target function was found in this node's subtree.

    """
    if isinstance(node, (cst.Import, cst.ImportFrom)):
        return None, False

    if isinstance(node, cst.FunctionDef):
        qualified_name = f"{prefix}.{node.name.value}" if prefix else node.name.value

        # Check if it's a helper of helper function
        if qualified_name in helpers_of_helper_functions:
            if remove_docstrings and isinstance(node.body, cst.IndentedBlock):
                return node.with_changes(body=remove_docstring_from_body(node.body)), True
            return node, True

        # Check if it's a target function
        if qualified_name in target_functions:
            if include_target_in_output:
                if remove_docstrings and isinstance(node.body, cst.IndentedBlock):
                    return node.with_changes(body=remove_docstring_from_body(node.body)), True
                return node, True
            return None, True

        # Check dunder methods
        # For read-only mode, exclude __init__; for testgen mode, include all dunders
        if is_dunder_method(node.name.value) and (include_init_dunder or node.name.value != "__init__"):
            if remove_docstrings and isinstance(node.body, cst.IndentedBlock):
                return node.with_changes(body=remove_docstring_from_body(node.body)), False
            return node, False

        return None, False

    if isinstance(node, cst.ClassDef):
        # Do not recurse into nested classes
        if prefix:
            return None, False
        # Assuming always an IndentedBlock
        if not isinstance(node.body, cst.IndentedBlock):
            raise ValueError("ClassDef body is not an IndentedBlock")  # noqa: TRY004

        class_prefix = f"{prefix}.{node.name.value}" if prefix else node.name.value

        # First pass: detect if there is a target function in the class
        found_in_class = False
        new_class_body: list[CSTNode] = []
        for stmt in node.body.body:
            filtered, found_target = prune_cst_for_context(
                stmt,
                target_functions,
                helpers_of_helper_functions,
                class_prefix,
                remove_docstrings=remove_docstrings,
                include_target_in_output=include_target_in_output,
                include_init_dunder=include_init_dunder,
            )
            found_in_class |= found_target
            if filtered:
                new_class_body.append(filtered)

        if not found_in_class:
            return None, False

        if remove_docstrings:
            return node.with_changes(
                body=remove_docstring_from_body(node.body.with_changes(body=new_class_body))
            ) if new_class_body else None, True
        return node.with_changes(body=node.body.with_changes(body=new_class_body)) if new_class_body else None, True

    # For other nodes, keep the node and recursively filter children
    section_names = get_section_names(node)
    if not section_names:
        return node, False

    updates: dict[str, list[cst.CSTNode] | cst.CSTNode] = {}
    found_any_target = False

    for section in section_names:
        original_content = getattr(node, section, None)
        if isinstance(original_content, (list, tuple)):
            new_children = []
            section_found_target = False
            for child in original_content:
                filtered, found_target = prune_cst_for_context(
                    child,
                    target_functions,
                    helpers_of_helper_functions,
                    prefix,
                    remove_docstrings=remove_docstrings,
                    include_target_in_output=include_target_in_output,
                    include_init_dunder=include_init_dunder,
                )
                if filtered:
                    new_children.append(filtered)
                section_found_target |= found_target

            if section_found_target or new_children:
                found_any_target |= section_found_target
                updates[section] = new_children
        elif original_content is not None:
            filtered, found_target = prune_cst_for_context(
                original_content,
                target_functions,
                helpers_of_helper_functions,
                prefix,
                remove_docstrings=remove_docstrings,
                include_target_in_output=include_target_in_output,
                include_init_dunder=include_init_dunder,
            )
            found_any_target |= found_target
            if filtered:
                updates[section] = filtered

    if updates:
        return (node.with_changes(**updates), found_any_target)

    return None, False
