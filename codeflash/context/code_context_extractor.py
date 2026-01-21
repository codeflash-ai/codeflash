from __future__ import annotations

import ast
import hashlib
import os
import sys
from collections import defaultdict
from itertools import chain
from typing import TYPE_CHECKING, cast

import jedi
import libcst as cst

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_extractor import add_needed_imports_from_module, find_preexisting_objects
from codeflash.code_utils.code_utils import encoded_tokens_len, get_qualified_name, path_belongs_to_site_packages
from codeflash.code_utils.compat import get_python_executable, is_compiled_or_bundled_binary
from codeflash.code_utils.config_consts import OPTIMIZATION_CONTEXT_TOKEN_LIMIT, TESTGEN_CONTEXT_TOKEN_LIMIT
from codeflash.code_utils.jedi_compat import safe_jedi_executable
from codeflash.context.unused_definition_remover import (
    collect_top_level_defs_with_usages,
    extract_names_from_targets,
    remove_unused_definitions_by_function_names,
)
from codeflash.discovery.functions_to_optimize import FunctionToOptimize  # noqa: TC001
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
    from libcst import CSTNode

    from codeflash.context.unused_definition_remover import UsageInfo


def _get_jedi_environment():
    """Get the appropriate jedi environment based on execution mode."""
    if is_compiled_or_bundled_binary():
        try:
            # In compiled mode, use InterpreterEnvironment to avoid subprocess spawning
            from jedi.api.environment import InterpreterEnvironment
            from pathlib import Path

            logger.info("[_get_jedi_environment] Creating InterpreterEnvironment for compiled binary")

            # Create InterpreterEnvironment which uses current process info
            env = InterpreterEnvironment()

            # Patch the executable and path to point to real Python instead of compiled binary
            # This allows jedi to find typeshed and other resources
            real_python_executable = get_python_executable()
            real_python_path = Path(real_python_executable)

            # Get the real sys.prefix (venv root or Python installation)
            if '.venv' in real_python_executable:
                # For venv: /path/to/.venv/bin/python3 -> /path/to/.venv
                real_sys_prefix = str(real_python_path.parent.parent)
            else:
                # For system Python: /usr/bin/python3 -> /usr
                real_sys_prefix = str(real_python_path.parent.parent)

            env.executable = real_python_executable
            env._start_executable = real_python_executable
            env.path = real_sys_prefix

            logger.info(f"[_get_jedi_environment] Created: executable={env.executable}, path={env.path}")
            return env
        except Exception as e:
            logger.error(f"[_get_jedi_environment] Failed to create InterpreterEnvironment: {e}", exc_info=True)
            # Return None to fall back to default behavior
            return None

    return None  # Let jedi auto-detect in normal mode


def _create_jedi_project(path: Path, **kwargs) -> jedi.Project:
    """Create a jedi.Project with the correct Python environment."""
    if is_compiled_or_bundled_binary():
        # In compiled mode: provide sys_path but avoid environment detection
        if 'sys_path' not in kwargs:
            kwargs['sys_path'] = list(sys.path)
        kwargs['smart_sys_path'] = False
        # Don't set environment_path - we'll pass environment directly to Script
        return jedi.Project(path=path, **kwargs)

    # When not compiled, use the detected Python executable normally
    python_executable = get_python_executable()
    if 'sys_path' not in kwargs:
        kwargs['sys_path'] = list(sys.path)
    return jedi.Project(path=path, environment_path=python_executable, **kwargs)


def get_code_optimization_context(
    function_to_optimize: FunctionToOptimize,
    project_root_path: Path,
    optim_token_limit: int = OPTIMIZATION_CONTEXT_TOKEN_LIMIT,
    testgen_token_limit: int = TESTGEN_CONTEXT_TOKEN_LIMIT,
) -> CodeOptimizationContext:
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

    # Extract code context for testgen
    testgen_context = extract_code_markdown_context_from_files(
        helpers_of_fto_dict,
        helpers_of_helpers_dict,
        project_root_path,
        remove_docstrings=False,
        code_context_type=CodeContextType.TESTGEN,
    )

    # Extract class definitions for imported types from project modules
    # This helps the LLM understand class constructors and structure
    imported_class_context = get_imported_class_definitions(testgen_context, project_root_path)
    if imported_class_context.code_strings:
        # Merge imported class definitions into testgen context
        testgen_context = CodeStringsMarkdown(
            code_strings=testgen_context.code_strings + imported_class_context.code_strings
        )

    testgen_markdown_code = testgen_context.markdown
    testgen_code_token_length = encoded_tokens_len(testgen_markdown_code)
    if testgen_code_token_length > testgen_token_limit:
        # First try removing docstrings
        testgen_context = extract_code_markdown_context_from_files(
            helpers_of_fto_dict,
            helpers_of_helpers_dict,
            project_root_path,
            remove_docstrings=True,
            code_context_type=CodeContextType.TESTGEN,
        )
        # Re-extract imported classes (they may still fit)
        imported_class_context = get_imported_class_definitions(testgen_context, project_root_path)
        if imported_class_context.code_strings:
            testgen_context = CodeStringsMarkdown(
                code_strings=testgen_context.code_strings + imported_class_context.code_strings
            )
        testgen_markdown_code = testgen_context.markdown
        testgen_code_token_length = encoded_tokens_len(testgen_markdown_code)
        if testgen_code_token_length > testgen_token_limit:
            # If still over limit, try without imported class definitions
            testgen_context = extract_code_markdown_context_from_files(
                helpers_of_fto_dict,
                helpers_of_helpers_dict,
                project_root_path,
                remove_docstrings=True,
                code_context_type=CodeContextType.TESTGEN,
            )
            testgen_markdown_code = testgen_context.markdown
            testgen_code_token_length = encoded_tokens_len(testgen_markdown_code)
            if testgen_code_token_length > testgen_token_limit:
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


def extract_code_string_context_from_files(
    helpers_of_fto: dict[Path, set[FunctionSource]],
    helpers_of_helpers: dict[Path, set[FunctionSource]],
    project_root_path: Path,
    remove_docstrings: bool = False,  # noqa: FBT001, FBT002
    code_context_type: CodeContextType = CodeContextType.READ_ONLY,
) -> CodeString:
    """Extract code context from files containing target functions and their helpers.
    This function processes two sets of files:
    1. Files containing the function to optimize (fto) and their first-degree helpers
    2. Files containing only helpers of helpers (with no overlap with the first set).

    For each file, it extracts relevant code based on the specified context type, adds necessary
    imports, and combines them.

    Args:
    ----
        helpers_of_fto: Dictionary mapping file paths to sets of Function Sources of function to optimize and its helpers
        helpers_of_helpers: Dictionary mapping file paths to sets of Function Sources of helpers of helper functions
        project_root_path: Root path of the project
        remove_docstrings: Whether to remove docstrings from the extracted code
        code_context_type: Type of code context to extract (READ_ONLY, READ_WRITABLE, or TESTGEN)

    Returns:
    -------
        CodeString containing the extracted code context with necessary imports

    """  # noqa: D205
    # Rearrange to remove overlaps, so we only access each file path once
    helpers_of_helpers_no_overlap = defaultdict(set)
    for file_path, function_sources in helpers_of_helpers.items():
        if file_path in helpers_of_fto:
            # Remove duplicates within the same file path, in case a helper of helper is also a helper of fto
            helpers_of_helpers[file_path] -= helpers_of_fto[file_path]
        else:
            helpers_of_helpers_no_overlap[file_path] = function_sources

    final_code_string_context = ""

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
            final_code_string_context += f"\n{code_context}"
            final_code_string_context = add_needed_imports_from_module(
                src_module_code=original_code,
                dst_module_code=final_code_string_context,
                src_path=file_path,
                dst_path=file_path,
                project_root=project_root_path,
                helper_functions=list(helpers_of_fto.get(file_path, set()) | helpers_of_helpers.get(file_path, set())),
            )
    if code_context_type == CodeContextType.READ_WRITABLE:
        return CodeString(code=final_code_string_context)
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
            final_code_string_context += f"\n{code_context}"
            final_code_string_context = add_needed_imports_from_module(
                src_module_code=original_code,
                dst_module_code=final_code_string_context,
                src_path=file_path,
                dst_path=file_path,
                project_root=project_root_path,
                helper_functions=list(helpers_of_helpers_no_overlap.get(file_path, set())),
            )
    return CodeString(code=final_code_string_context)


def extract_code_markdown_context_from_files(
    helpers_of_fto: dict[Path, set[FunctionSource]],
    helpers_of_helpers: dict[Path, set[FunctionSource]],
    project_root_path: Path,
    remove_docstrings: bool = False,  # noqa: FBT001, FBT002
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

    try:
        # IMPORTANT: Get jedi environment BEFORE entering safe_jedi_executable context
        # because safe_jedi_executable patches sys.executable which affects is_compiled_or_bundled_binary()
        jedi_env = _get_jedi_environment()

        with safe_jedi_executable():
            # Use jedi to find function to optimize
            # Pass environment directly to Script to avoid subprocess spawning in compiled mode
            script = jedi.Script(
                path=function_to_optimize.file_path,
                project=_create_jedi_project(project_root_path),
                environment=jedi_env
            )

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
    except Exception as e:
        # If jedi completely fails, provide a more helpful error message
        if is_compiled_or_bundled_binary():
            logger.error(f"Jedi failed to analyze {function_to_optimize.file_path} in compiled mode: {e}")
        raise ValueError(
            f"Could not analyze function {function_to_optimize.function_name} in {function_to_optimize.file_path}: {e}"
        ) from e


def get_function_sources_from_jedi(
    file_path_to_qualified_function_names: dict[Path, set[str]], project_root_path: Path
) -> tuple[dict[Path, set[FunctionSource]], list[FunctionSource]]:
    import jedi

    file_path_to_function_source = defaultdict(set)
    function_source_list: list[FunctionSource] = []

    # IMPORTANT: Get jedi environment BEFORE entering safe_jedi_executable context
    # because safe_jedi_executable patches sys.executable which affects is_compiled_or_bundled_binary()
    jedi_env = _get_jedi_environment()

    with safe_jedi_executable():
        for file_path, qualified_function_names in file_path_to_qualified_function_names.items():
            try:
                # Pass environment directly to Script to avoid subprocess spawning in compiled mode
                script = jedi.Script(
                    path=file_path,
                    project=_create_jedi_project(project_root_path),
                    environment=jedi_env
                )
                file_refs = script.get_names(all_scopes=True, definitions=False, references=True)
            except Exception as e:
                # Jedi can fail in compiled binaries despite mitigation
                if is_compiled_or_bundled_binary():
                    logger.warning(f"Jedi analysis failed in compiled mode for {file_path}, skipping: {e}")
                    continue
                else:
                    # In normal mode, re-raise to surface the real issue
                    raise

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
                    imported_name = alias.asname if alias.asname else alias.name
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

    # IMPORTANT: Get jedi environment BEFORE entering safe_jedi_executable context
    # because safe_jedi_executable patches sys.executable which affects is_compiled_or_bundled_binary()
    jedi_env = _get_jedi_environment()

    with safe_jedi_executable():
        for name, module_name in imported_names.items():
            # Skip if already defined in context
            if name in existing_definitions:
                continue

            # Try to find the module file using Jedi
            try:
                # Create a script that imports the module to resolve it
                test_code = f"import {module_name}"
                # Pass environment directly to Script to avoid subprocess spawning in compiled mode
                script = jedi.Script(
                    test_code,
                    project=_create_jedi_project(project_root_path),
                    environment=jedi_env
                )
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

                # Skip if we've already extracted this class
                if (module_path, name) in extracted_classes:
                    continue

                # Parse the module to find the class definition
                module_source = module_path.read_text(encoding="utf-8")
                module_tree = ast.parse(module_source)

                for node in ast.walk(module_tree):
                    if isinstance(node, ast.ClassDef) and node.name == name:
                        # Extract the class source code
                        lines = module_source.split("\n")
                        class_source = "\n".join(lines[node.lineno - 1 : node.end_lineno])

                        # Also extract any necessary imports for the class (base classes, type hints)
                        class_imports = _extract_imports_for_class(module_tree, node, module_source)

                        full_source = class_imports + "\n\n" + class_source if class_imports else class_source

                        class_code_strings.append(CodeString(code=full_source, file_path=module_path))
                        extracted_classes.add((module_path, name))
                        break

            except Exception:
                logger.debug(f"Error extracting class definition for {name} from {module_name}")
                continue

    return CodeStringsMarkdown(code_strings=class_code_strings)


def _extract_imports_for_class(module_tree: ast.Module, class_node: ast.ClassDef, module_source: str) -> str:
    """Extract import statements needed for a class definition.

    This extracts imports for base classes and commonly used type annotations.
    """
    needed_names: set[str] = set()

    # Get base class names
    for base in class_node.bases:
        if isinstance(base, ast.Name):
            needed_names.add(base.id)
        elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
            # For things like abc.ABC, we need the module name
            needed_names.add(base.value.id)

    # Find imports that provide these names
    import_lines: list[str] = []
    source_lines = module_source.split("\n")

    for node in module_tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name.split(".")[0]
                if name in needed_names:
                    import_lines.append(source_lines[node.lineno - 1])
                    break
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                if name in needed_names:
                    import_lines.append(source_lines[node.lineno - 1])
                    break

    return "\n".join(import_lines)


def is_dunder_method(name: str) -> bool:
    return len(name) > 4 and name.isascii() and name.startswith("__") and name.endswith("__")


def get_section_names(node: cst.CSTNode) -> list[str]:
    """Returns the section attribute names (e.g., body, orelse) for a given node if they exist."""  # noqa: D401
    possible_sections = ["body", "orelse", "finalbody", "handlers"]
    return [sec for sec in possible_sections if hasattr(node, sec)]


def remove_docstring_from_body(indented_block: cst.IndentedBlock) -> cst.CSTNode:
    """Removes the docstring from an indented block if it exists."""  # noqa: D401
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
    remove_docstrings: bool = False,  # noqa: FBT001, FBT002
) -> str:
    """Create a read-only version of the code by parsing and filtering the code to keep only class contextual information, and other module scoped variables."""
    module = cst.parse_module(code)
    defs_with_usages = collect_top_level_defs_with_usages(module, target_functions | helpers_of_helper_functions)

    if code_context_type == CodeContextType.READ_WRITABLE:
        filtered_node, found_target = prune_cst_for_read_writable_code(module, target_functions, defs_with_usages)
    elif code_context_type == CodeContextType.READ_ONLY:
        filtered_node, found_target = prune_cst_for_read_only_code(
            module, target_functions, helpers_of_helper_functions, remove_docstrings=remove_docstrings
        )
    elif code_context_type == CodeContextType.TESTGEN:
        filtered_node, found_target = prune_cst_for_testgen_code(
            module, target_functions, helpers_of_helper_functions, remove_docstrings=remove_docstrings
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


def prune_cst_for_read_writable_code(  # noqa: PLR0911
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
        # Assuming always an IndentedBlock
        if not isinstance(node.body, cst.IndentedBlock):
            raise ValueError("ClassDef body is not an IndentedBlock")  # noqa: TRY004
        class_prefix = f"{prefix}.{node.name.value}" if prefix else node.name.value
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


def prune_cst_for_code_hashing(  # noqa: PLR0911
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


def prune_cst_for_read_only_code(  # noqa: PLR0911
    node: cst.CSTNode,
    target_functions: set[str],
    helpers_of_helper_functions: set[str],
    prefix: str = "",
    remove_docstrings: bool = False,  # noqa: FBT001, FBT002
) -> tuple[cst.CSTNode | None, bool]:
    """Recursively filter the node for read-only context.

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
        # If it's a target function, remove it but mark found_target = True
        if qualified_name in helpers_of_helper_functions:
            return node, True
        if qualified_name in target_functions:
            return None, True
        # Keep only dunder methods
        if is_dunder_method(node.name.value) and node.name.value != "__init__":
            if remove_docstrings and isinstance(node.body, cst.IndentedBlock):
                new_body = remove_docstring_from_body(node.body)
                return node.with_changes(body=new_body), False
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
            filtered, found_target = prune_cst_for_read_only_code(
                stmt, target_functions, helpers_of_helper_functions, class_prefix, remove_docstrings=remove_docstrings
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
                filtered, found_target = prune_cst_for_read_only_code(
                    child, target_functions, helpers_of_helper_functions, prefix, remove_docstrings=remove_docstrings
                )
                if filtered:
                    new_children.append(filtered)
                section_found_target |= found_target

            if section_found_target or new_children:
                found_any_target |= section_found_target
                updates[section] = new_children
        elif original_content is not None:
            filtered, found_target = prune_cst_for_read_only_code(
                original_content,
                target_functions,
                helpers_of_helper_functions,
                prefix,
                remove_docstrings=remove_docstrings,
            )
            found_any_target |= found_target
            if filtered:
                updates[section] = filtered
    if updates:
        return (node.with_changes(**updates), found_any_target)

    return None, False


def prune_cst_for_testgen_code(  # noqa: PLR0911
    node: cst.CSTNode,
    target_functions: set[str],
    helpers_of_helper_functions: set[str],
    prefix: str = "",
    remove_docstrings: bool = False,  # noqa: FBT001, FBT002
) -> tuple[cst.CSTNode | None, bool]:
    """Recursively filter the node for testgen context.

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
        # If it's a target function, remove it but mark found_target = True
        if qualified_name in helpers_of_helper_functions or qualified_name in target_functions:
            if remove_docstrings and isinstance(node.body, cst.IndentedBlock):
                new_body = remove_docstring_from_body(node.body)
                return node.with_changes(body=new_body), True
            return node, True
        # Keep all dunder methods
        if is_dunder_method(node.name.value):
            if remove_docstrings and isinstance(node.body, cst.IndentedBlock):
                new_body = remove_docstring_from_body(node.body)
                return node.with_changes(body=new_body), False
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
            filtered, found_target = prune_cst_for_testgen_code(
                stmt, target_functions, helpers_of_helper_functions, class_prefix, remove_docstrings=remove_docstrings
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
                filtered, found_target = prune_cst_for_testgen_code(
                    child, target_functions, helpers_of_helper_functions, prefix, remove_docstrings=remove_docstrings
                )
                if filtered:
                    new_children.append(filtered)
                section_found_target |= found_target

            if section_found_target or new_children:
                found_any_target |= section_found_target
                updates[section] = new_children
        elif original_content is not None:
            filtered, found_target = prune_cst_for_testgen_code(
                original_content,
                target_functions,
                helpers_of_helper_functions,
                prefix,
                remove_docstrings=remove_docstrings,
            )
            found_any_target |= found_target
            if filtered:
                updates[section] = filtered
    if updates:
        return (node.with_changes(**updates), found_any_target)

    return None, False
