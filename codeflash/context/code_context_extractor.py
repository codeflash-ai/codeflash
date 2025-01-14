from __future__ import annotations

import os
from collections import defaultdict
from itertools import chain
from pathlib import Path

import jedi
import libcst as cst
import tiktoken
from jedi.api.classes import Name
from libcst import CSTNode

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_extractor import add_needed_imports_from_module, find_preexisting_objects
from codeflash.code_utils.code_utils import get_qualified_name, path_belongs_to_site_packages
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import CodeOptimizationContext, CodeString, CodeStringsMarkdown, FunctionSource
from codeflash.optimization.function_context import belongs_to_function_qualified


def get_code_optimization_context(
    function_to_optimize: FunctionToOptimize, project_root_path: Path, token_limit: int = 8000
) -> CodeOptimizationContext:
    # Get qualified names and fully qualified names(fqn) of helpers
    helpers_of_fto, helpers_of_fto_fqn, helpers_of_fto_obj_list = get_file_path_to_helper_functions_dict(
        {function_to_optimize.file_path: {function_to_optimize.qualified_name}}, project_root_path
    )

    helpers_of_helpers, helpers_of_helpers_fqn, _ = get_file_path_to_helper_functions_dict(
        helpers_of_fto, project_root_path
    )
    print("helpers_of_fto")
    print(helpers_of_fto)
    print("helpers_of_helpers")
    print(helpers_of_helpers)
    # Add function to optimize
    helpers_of_fto[function_to_optimize.file_path].add(function_to_optimize.qualified_name)
    helpers_of_fto_fqn[function_to_optimize.file_path].add(
        function_to_optimize.qualified_name_with_modules_from_root(project_root_path)
    )

    # Extract code
    final_read_writable_code = get_all_read_writable_code(helpers_of_fto, helpers_of_fto_fqn, project_root_path).code
    read_only_code_markdown = get_all_read_only_code_context(
        helpers_of_fto,
        helpers_of_fto_fqn,
        helpers_of_helpers,
        helpers_of_helpers_fqn,
        project_root_path,
        remove_docstrings=False,
    )
    testgen_context_code = get_all_testgen_context(helpers_of_fto, helpers_of_fto_fqn, project_root_path)

    # Handle token limits
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    final_read_writable_tokens = len(tokenizer.encode(final_read_writable_code))
    if final_read_writable_tokens > token_limit:
        raise ValueError("Read-writable code has exceeded token limit, cannot proceed")
    # if len(tokenizer.encode(testgen_context_code.code)) > token_limit:
    #     raise ValueError("Testgen context has exceeded token limit, cannot proceed")
    # Setup preexisting objects for code replacer TODO: should remove duplicates
    preexisting_objects = list(
        chain(
            find_preexisting_objects(final_read_writable_code),
            *(find_preexisting_objects(codestring.code) for codestring in read_only_code_markdown.code_strings),
        )
    )
    read_only_code_markdown_tokens = len(tokenizer.encode(read_only_code_markdown.markdown))
    total_tokens = final_read_writable_tokens + read_only_code_markdown_tokens
    if total_tokens <= token_limit:
        return CodeOptimizationContext(
            code_to_optimize_with_helpers="",
            read_writable_code=CodeString(code=final_read_writable_code).code,
            read_only_context_code=read_only_code_markdown.markdown,
            helper_functions=helpers_of_fto_obj_list,
            preexisting_objects=preexisting_objects,
            testgen_context_code=testgen_context_code.code,
        )

    logger.debug("Code context has exceeded token limit, removing docstrings from read-only code")

    # Extract read only code without docstrings
    read_only_code_no_docstring_markdown = get_all_read_only_code_context(
        helpers_of_fto,
        helpers_of_fto_fqn,
        helpers_of_helpers,
        helpers_of_helpers_fqn,
        project_root_path,
        remove_docstrings=True,
    )
    read_only_code_no_docstring_markdown_tokens = len(tokenizer.encode(read_only_code_no_docstring_markdown.markdown))
    total_tokens = final_read_writable_tokens + read_only_code_no_docstring_markdown_tokens
    if total_tokens <= token_limit:
        return CodeOptimizationContext(
            code_to_optimize_with_helpers="",
            read_writable_code=CodeString(code=final_read_writable_code).code,
            read_only_context_code=read_only_code_no_docstring_markdown.markdown,
            helper_functions=helpers_of_fto_obj_list,
            preexisting_objects=preexisting_objects,
            testgen_context_code=testgen_context_code.code,
        )

    logger.debug("Code context has exceeded token limit, removing read-only code")
    return CodeOptimizationContext(
        code_to_optimize_with_helpers="",
        read_writable_code=CodeString(code=final_read_writable_code).code,
        read_only_context_code="",
        helper_functions=helpers_of_fto_obj_list,
        preexisting_objects=preexisting_objects,
        testgen_context_code=testgen_context_code.code,
    )


def get_all_testgen_context(
    helpers_of_fto: dict[Path, set[str]], helpers_of_fto_fqn: dict[Path, set[str]], project_root_path: Path
) -> CodeString:
    final_testgen_context = ""
    # Extract code from file paths that contain fto and first degree helpers. helpers of helpers may also be included if they are in the same files
    for file_path, qualified_function_names in helpers_of_fto.items():
        try:
            original_code = file_path.read_text("utf8")
        except Exception as e:
            logger.exception(f"Error while parsing {file_path}: {e}")
            continue
        try:
            testgen_context_code = get_testgen_context(original_code, qualified_function_names)
        except ValueError as e:
            logger.debug(f"Error while getting read-only code: {e}")
            continue

        if testgen_context_code:
            final_testgen_context += f"\n{testgen_context_code}"
            final_testgen_context = add_needed_imports_from_module(
                src_module_code=original_code,
                dst_module_code=final_testgen_context,
                src_path=file_path,
                dst_path=file_path,
                project_root=project_root_path,
                helper_functions_fqn=helpers_of_fto_fqn[file_path],
            )
    return CodeString(code=final_testgen_context)


def get_all_read_writable_code(
    helpers_of_fto: dict[Path, set[str]], helpers_of_fto_fqn: dict[Path, set[str]], project_root_path: Path
) -> CodeString:
    final_read_writable_code = ""
    # Extract code from file paths that contain fto and first degree helpers
    for file_path, qualified_function_names in helpers_of_fto.items():
        try:
            original_code = file_path.read_text("utf8")
        except Exception as e:
            logger.exception(f"Error while parsing {file_path}: {e}")
            continue
        try:
            read_writable_code = get_read_writable_code(original_code, qualified_function_names)
        except ValueError as e:
            logger.debug(f"Error while getting read-writable code: {e}")
            continue

        if read_writable_code:
            final_read_writable_code += f"\n{read_writable_code}"
            final_read_writable_code = add_needed_imports_from_module(
                src_module_code=original_code,
                dst_module_code=final_read_writable_code,
                src_path=file_path,
                dst_path=file_path,
                project_root=project_root_path,
                helper_functions_fqn=helpers_of_fto_fqn[file_path],
            )
    return CodeString(code=final_read_writable_code)


def get_all_read_only_code_context(
    helpers_of_fto: dict[Path, set[str]],
    helpers_of_fto_fqn: dict[Path, set[str]],
    helpers_of_helpers: dict[Path, set[str]],
    helpers_of_helpers_fqn: dict[Path, set[str]],
    project_root_path: Path,
    remove_docstrings: bool = False,
) -> CodeStringsMarkdown:
    # Rearrange to remove overlaps, so we only access each file path once
    helpers_of_helpers_no_overlap = defaultdict(set)
    helpers_of_helpers_no_overlap_fqn = defaultdict(set)
    for file_path in helpers_of_helpers:
        if file_path in helpers_of_fto:
            # Remove duplicates, in case a helper of helper is also a helper of fto
            helpers_of_helpers[file_path] -= helpers_of_fto[file_path]
            helpers_of_helpers_fqn[file_path] -= helpers_of_fto_fqn[file_path]
        else:
            helpers_of_helpers_no_overlap[file_path] = helpers_of_helpers[file_path]
            helpers_of_helpers_no_overlap_fqn[file_path] = helpers_of_helpers_fqn[file_path]

    read_only_code_markdown = CodeStringsMarkdown()
    # Extract code from file paths that contain fto and first degree helpers. helpers of helpers may also be included if they are in the same files
    for file_path, qualified_function_names in helpers_of_fto.items():
        try:
            original_code = file_path.read_text("utf8")
        except Exception as e:
            logger.exception(f"Error while parsing {file_path}: {e}")
            continue
        try:
            read_only_code = get_read_only_code(
                original_code, qualified_function_names, helpers_of_helpers.get(file_path, set()), remove_docstrings
            )
        except ValueError as e:
            logger.debug(f"Error while getting read-only code: {e}")
            continue
        if read_only_code.strip():
            read_only_code_with_imports = CodeString(
                code=add_needed_imports_from_module(
                    src_module_code=original_code,
                    dst_module_code=read_only_code,
                    src_path=file_path,
                    dst_path=file_path,
                    project_root=project_root_path,
                    helper_functions_fqn=helpers_of_fto_fqn[file_path] | helpers_of_helpers_fqn[file_path],
                ),
                file_path=file_path.relative_to(project_root_path),
            )
            read_only_code_markdown.code_strings.append(read_only_code_with_imports)

    # Extract code from file paths containing helpers of helpers
    for file_path, qualified_helper_function_names in helpers_of_helpers_no_overlap.items():
        try:
            original_code = file_path.read_text("utf8")
        except Exception as e:
            logger.exception(f"Error while parsing {file_path}: {e}")
            continue
        try:
            read_only_code = get_read_only_code(
                original_code, set(), qualified_helper_function_names, remove_docstrings
            )
        except ValueError as e:
            logger.debug(f"Error while getting read-only code: {e}")
            continue

        if read_only_code.strip():
            read_only_code_with_imports = CodeString(
                code=add_needed_imports_from_module(
                    src_module_code=original_code,
                    dst_module_code=read_only_code,
                    src_path=file_path,
                    dst_path=file_path,
                    project_root=project_root_path,
                    helper_functions_fqn=helpers_of_helpers_no_overlap_fqn[file_path],
                ),
                file_path=file_path.relative_to(project_root_path),
            )
            read_only_code_markdown.code_strings.append(read_only_code_with_imports)
    return read_only_code_markdown


def get_file_path_to_helper_functions_dict(
    file_path_to_qualified_function_names: dict[Path, set[str]], project_root_path: Path
) -> tuple[dict[Path, set[str]], dict[Path, set[str]], list[FunctionSource]]:
    file_path_to_helper_function_qualified_names = defaultdict(set)
    file_path_to_helper_function_fqn = defaultdict(set)
    function_source_list: list[FunctionSource] = []
    for file_path in file_path_to_qualified_function_names:
        script = jedi.Script(path=file_path, project=jedi.Project(path=project_root_path))
        file_refs = script.get_names(all_scopes=True, definitions=False, references=True)

        for qualified_function_name in file_path_to_qualified_function_names[file_path]:
            names = [
                ref
                for ref in file_refs
                if ref.full_name and belongs_to_function_qualified(ref, qualified_function_name)
            ]
            for name in names:
                try:
                    definitions: list[Name] = name.goto(follow_imports=True, follow_builtin_imports=False)
                except Exception as e:
                    try:
                        logger.exception(f"Error while getting definition for {name.full_name}: {e}")
                    except Exception as e:
                        # name.full_name can also throw exceptions sometimes
                        logger.exception(f"Error while getting definition: {e}")
                    definitions = []
                if definitions:
                    # TODO: there can be multiple definitions, see how to handle such cases
                    definition = definitions[0]
                    definition_path = definition.module_path

                    # The definition is part of this project and not defined within the original function
                    if (
                        str(definition_path).startswith(str(project_root_path) + os.sep)
                        and not path_belongs_to_site_packages(definition_path)
                        and definition.full_name
                        and definition.type == "function"
                        and not belongs_to_function_qualified(definition, qualified_function_name)
                    ):
                        file_path_to_helper_function_qualified_names[definition_path].add(
                            get_qualified_name(definition.module_name, definition.full_name)
                        )
                        file_path_to_helper_function_fqn[definition_path].add(definition.full_name)
                        function_source_list.append(
                            FunctionSource(
                                file_path=definition_path,
                                qualified_name=get_qualified_name(definition.module_name, definition.full_name),
                                fully_qualified_name=definition.full_name,
                                only_function_name=definition.name,
                                source_code=definition.get_line_code(),
                                jedi_definition=definition,
                            )
                        )

    return file_path_to_helper_function_qualified_names, file_path_to_helper_function_fqn, function_source_list


def is_dunder_method(name: str) -> bool:
    return len(name) > 4 and name.isascii() and name.startswith("__") and name.endswith("__")


def get_section_names(node: cst.CSTNode) -> list[str]:
    """Returns the section attribute names (e.g., body, orelse) for a given node if they exist."""
    possible_sections = ["body", "orelse", "finalbody", "handlers"]
    return [sec for sec in possible_sections if hasattr(node, sec)]


def remove_docstring_from_body(indented_block: cst.IndentedBlock) -> cst.CSTNode:
    """Removes the docstring from an indented block if it exists"""
    if not isinstance(indented_block.body[0], cst.SimpleStatementLine):
        return indented_block
    first_stmt = indented_block.body[0].body[0]
    if isinstance(first_stmt, cst.Expr) and isinstance(first_stmt.value, cst.SimpleString):
        return indented_block.with_changes(body=indented_block.body[1:])
    return indented_block


def prune_cst_for_read_writable_code(
    node: cst.CSTNode, target_functions: set[str], prefix: str = ""
) -> tuple[cst.CSTNode | None, bool]:
    """Recursively filter the node and its children to build the read-writable codeblock. This contains nodes that lead to target functions.

    Returns:
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
            raise ValueError("ClassDef body is not an IndentedBlock")
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
                filtered, found_target = prune_cst_for_read_writable_code(child, target_functions, prefix)
                if filtered:
                    new_children.append(filtered)
                section_found_target |= found_target

            if section_found_target:
                found_any_target = True
                updates[section] = new_children
        elif original_content is not None:
            filtered, found_target = prune_cst_for_read_writable_code(original_content, target_functions, prefix)
            if found_target:
                found_any_target = True
                if filtered:
                    updates[section] = filtered

    if not found_any_target:
        return None, False

    return (node.with_changes(**updates) if updates else node), True


def get_read_writable_code(code: str, target_functions: set[str]) -> str:
    """Creates a read-writable code string by parsing and filtering the code to keep only
    target functions and the minimal surrounding structure.
    """
    module = cst.parse_module(code)
    filtered_node, found_target = prune_cst_for_read_writable_code(module, target_functions)
    if not found_target:
        raise ValueError("No target functions found in the provided code")
    if filtered_node and isinstance(filtered_node, cst.Module):
        return str(filtered_node.code)
    return ""


def prune_cst_for_read_only_code(
    node: cst.CSTNode,
    target_functions: set[str],
    helpers_of_helper_functions: set[str],
    prefix: str = "",
    remove_docstrings: bool = False,
) -> tuple[cst.CSTNode | None, bool]:
    """Recursively filter the node for read-only context:

    Returns:
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
            raise ValueError("ClassDef body is not an IndentedBlock")

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


def get_read_only_code(
    code: str, target_functions: set[str], helpers_of_helper_functions: set[str], remove_docstrings: bool = False
) -> str:
    """Creates a read-only version of the code by parsing and filtering the code to keep only
    class contextual information, and other module scoped variables.
    """
    module = cst.parse_module(code)
    filtered_node, found_target = prune_cst_for_read_only_code(
        module, target_functions, helpers_of_helper_functions, remove_docstrings=remove_docstrings
    )
    if not found_target:
        raise ValueError("No target functions found in the provided code")
    if filtered_node and isinstance(filtered_node, cst.Module):
        return str(filtered_node.code)
    return ""


def prune_cst_for_testgen_context(
    node: cst.CSTNode, target_functions: set[str], prefix: str = "", remove_docstrings: bool = False
) -> tuple[cst.CSTNode | None, bool]:
    """Recursively filter the node for read-only context:

    Returns:
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
        # Keep only dunder methods
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
            raise ValueError("ClassDef body is not an IndentedBlock")

        class_prefix = f"{prefix}.{node.name.value}" if prefix else node.name.value

        # First pass: detect if there is a target function in the class
        found_in_class = False
        new_class_body: list[CSTNode] = []
        for stmt in node.body.body:
            filtered, found_target = prune_cst_for_testgen_context(
                stmt, target_functions, class_prefix, remove_docstrings=remove_docstrings
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
                filtered, found_target = prune_cst_for_testgen_context(
                    child, target_functions, prefix, remove_docstrings=remove_docstrings
                )
                if filtered:
                    new_children.append(filtered)
                section_found_target |= found_target

            if section_found_target or new_children:
                found_any_target |= section_found_target
                updates[section] = new_children
        elif original_content is not None:
            filtered, found_target = prune_cst_for_testgen_context(
                original_content, target_functions, prefix, remove_docstrings=remove_docstrings
            )
            found_any_target |= found_target
            if filtered:
                updates[section] = filtered
    if updates:
        return (node.with_changes(**updates), found_any_target)

    return None, False


def get_testgen_context(code: str, target_functions: set[str], remove_docstrings: bool = False) -> str:
    """Creates testgen_context. Similar to get_read_only_code, except the target functions are included."""
    module = cst.parse_module(code)
    filtered_node, found_target = prune_cst_for_testgen_context(
        module, target_functions, remove_docstrings=remove_docstrings
    )
    if not found_target:
        raise ValueError("No target functions found in the provided code")
    if filtered_node and isinstance(filtered_node, cst.Module):
        return str(filtered_node.code)
    return ""
