import os
from collections import defaultdict
from pathlib import Path

import jedi
from jedi.api.classes import Name

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_extractor import add_needed_imports_from_module
from codeflash.code_utils.code_utils import get_qualified_name, path_belongs_to_site_packages
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.optimization.cst_manipulator import get_read_only_code, get_read_writable_code
from codeflash.optimization.function_context import belongs_to_class, belongs_to_function


def get_code_optimization_context(
    function_to_optimize: FunctionToOptimize, project_root_path: Path, original_source_code: str
) -> [str, str]:
    function_name = function_to_optimize.function_name
    file_path = function_to_optimize.file_path
    script = jedi.Script(path=file_path, project=jedi.Project(path=project_root_path))
    file_path_to_qualified_function_names = defaultdict(set)
    file_path_to_qualified_function_names[file_path].add(function_to_optimize.qualified_name)
    read_only_list = []
    final_read_writable_code = ""
    final_read_only_code = ""
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
        except Exception as e:
            logger.exception(f"Error while parsing {file_path}: {e}")
            continue
        try:
            read_writable_code = get_read_writable_code(og_code_containing_helpers, qualified_function_names)
        except ValueError as e:
            logger.debug(f"Error while getting read-writable code: {e}")
            continue

        if read_writable_code:
            final_read_writable_code += f"\n{read_writable_code}"
            final_read_writable_code = add_needed_imports_from_module(
                src_module_code=og_code_containing_helpers,
                dst_module_code=final_read_writable_code,
                src_path=file_path,
                dst_path=file_path,
                project_root=project_root_path,
                helper_functions_fully_qualified_names=list(qualified_function_names),
            )

        try:
            read_only_code = get_read_only_code(og_code_containing_helpers, qualified_function_names)
        except ValueError as e:
            logger.debug(f"Error while getting read-only code: {e}")
            continue

        read_only_code_with_imports = add_needed_imports_from_module(
            src_module_code=og_code_containing_helpers,
            dst_module_code=read_only_code,
            src_path=file_path,
            dst_path=file_path,
            project_root=project_root_path,
            helper_functions_fully_qualified_names=list(qualified_function_names),
        )
        if read_only_code_with_imports:
            read_only_list.append(f"```python:{file_path}\n{read_only_code_with_imports}```")

    final_read_only_code = "\n".join(read_only_list)

    return final_read_writable_code, final_read_only_code
