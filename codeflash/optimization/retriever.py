import os
from collections import defaultdict
from pathlib import Path

import jedi
import libcst as cst
from jedi.api.classes import Name
from returns.result import Result

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_extractor import add_needed_imports_from_module_2
from codeflash.code_utils.code_utils import get_qualified_name, path_belongs_to_site_packages
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import CodeOptimizationContext
from codeflash.optimization.cst_context import (
    CSTContextNode,
    build_context_tree,
    create_read_only_context,
    create_read_write_context,
)
from codeflash.optimization.function_context import belongs_to_class, belongs_to_function


def get_code_optimization_context(
    function_to_optimize: FunctionToOptimize, project_root_path: Path, original_source_code: str
) -> Result[CodeOptimizationContext, str]:
    function_name = function_to_optimize.function_name
    file_path = function_to_optimize.file_path
    script = jedi.Script(path=file_path, project=jedi.Project(path=project_root_path))
    file_path_to_qualified_function_names = defaultdict(set)
    file_path_to_qualified_function_names[file_path].add(function_to_optimize.qualified_name)
    read_write_list = []
    read_only_list = []
    read_write_string = ""
    read_only_string = ""
    names = []
    for ref in script.get_names(all_scopes=True, definitions=False, references=True):
        if ref.full_name:
            if function_to_optimize.parents:
                # Check if the reference belongs to the specified class when FunctionParent is provided
                if belongs_to_class(ref, function_to_optimize.parents[-1].name) and belongs_to_function(
                    ref, function_name
                ):
                    names.append(ref)
            elif belongs_to_function(ref, function_name):
                names.append(ref)

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
                and not belongs_to_function(definition, function_name)
            ):
                file_path_to_qualified_function_names[definition_path].add(
                    get_qualified_name(definition.module_name, definition.full_name)
                )
    for file_path, qualified_function_names in file_path_to_qualified_function_names.items():
        try:
            og_code_containing_helpers = file_path.read_text("utf8")
            context_tree_root = cst.parse_module(og_code_containing_helpers)
        except Exception as e:
            logger.exception(f"Error while parsing {file_path}: {e}")
            continue
        context_tree_root = CSTContextNode(cst_node=context_tree_root, target_functions=qualified_function_names)
        if not build_context_tree(context_tree_root, ""):
            logger.debug(
                f"{qualified_function_names} was not found in {file_path} when retrieving code optimization context"
            )
            continue

        read_write_context_string = f"{create_read_write_context(context_tree_root)}"
        if read_write_context_string:
            read_write_string += f"\n{read_write_context_string}"
            read_write_string = add_needed_imports_from_module_2(
                og_code_containing_helpers,
                read_write_string,
                file_path,
                file_path,
                project_root_path,
                list(qualified_function_names),
            )

        read_only_context_string = f"{create_read_only_context(context_tree_root)}\n"
        read_only_context_string_with_imports = add_needed_imports_from_module_2(
            og_code_containing_helpers,
            read_only_context_string,
            file_path,
            file_path,
            project_root_path,
            list(qualified_function_names),
        )
        if read_only_context_string_with_imports:
            read_only_list.append(f"```python:{file_path}\n{read_only_context_string_with_imports}```")

    read_only_string = "\n".join(read_only_list)

    return read_write_string, read_only_string
