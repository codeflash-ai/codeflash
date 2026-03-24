from __future__ import annotations

import logging
import os
from collections import defaultdict
from functools import cache
from typing import TYPE_CHECKING

from codeflash.models.models import FunctionSource
from codeflash_python.code_utils.code_utils import path_belongs_to_site_packages
from codeflash_python.context.utils import get_qualified_name

if TYPE_CHECKING:
    from pathlib import Path

    from jedi.api.classes import Name

    from codeflash_core.models import FunctionToOptimize

logger = logging.getLogger("codeflash_python")


@cache
def get_jedi_project(project_root_path: str):  # noqa: ANN201
    import sys

    import jedi

    return jedi.Project(path=project_root_path, added_sys_path=list(sys.path))


def get_function_to_optimize_as_function_source(
    function_to_optimize: FunctionToOptimize, project_root_path: Path
) -> FunctionSource:
    import jedi

    # Use jedi to find function to optimize
    script = jedi.Script(path=function_to_optimize.file_path, project=get_jedi_project(str(project_root_path)))

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
                )
        except Exception as e:
            logger.exception("Error while getting function source: %s", e)
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
        script = jedi.Script(path=file_path, project=get_jedi_project(str(project_root_path)))
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
                    logger.debug("Error while getting definitions for %s", qualified_function_name)
                    definitions = []
                if definitions:
                    # TODO: there can be multiple definitions, see how to handle such cases
                    definition = definitions[0]
                    definition_path = definition.module_path
                    if definition_path is not None:
                        try:
                            rel = definition_path.resolve().relative_to(project_root_path.resolve())
                            definition_path = project_root_path / rel
                        except ValueError:
                            pass

                    # The definition is part of this project and not defined within the original function
                    is_valid_definition = (
                        definition_path is not None
                        and not path_belongs_to_site_packages(definition_path)
                        and str(definition_path).startswith(str(project_root_path) + os.sep)
                        and definition.full_name
                        and not belongs_to_function_qualified(definition, qualified_function_name)
                        and definition.full_name.startswith(definition.module_name)
                    )
                    if is_valid_definition and definition.type in ("function", "class", "statement"):
                        assert definition_path is not None
                        if definition.type == "function":
                            fqn = definition.full_name
                            func_name = definition.name
                        elif definition.type == "class":
                            fqn = f"{definition.full_name}.__init__"
                            func_name = "__init__"
                        else:
                            fqn = definition.full_name
                            func_name = definition.name
                        qualified_name = get_qualified_name(definition.module_name, fqn)
                        # Avoid self-references (recursive calls) and nested functions/classes
                        if qualified_name == qualified_function_name:
                            continue
                        if len(qualified_name.split(".")) <= 2:
                            function_source = FunctionSource(
                                file_path=definition_path,
                                qualified_name=qualified_name,
                                fully_qualified_name=fqn,
                                only_function_name=func_name,
                                source_code=definition.get_line_code(),
                                definition_type=definition.type,
                            )
                            file_path_to_function_source[definition_path].add(function_source)
                            function_source_list.append(function_source)

    return file_path_to_function_source, function_source_list


def belongs_to_method(name: Name, class_name: str, method_name: str) -> bool:
    """Check if the given name belongs to the specified method."""
    return belongs_to_function(name, method_name) and belongs_to_class(name, class_name)


def belongs_to_function(name: Name, function_name: str) -> bool:
    """Check if the given jedi Name is a direct child of the specified function."""
    if name.name == function_name:  # Handles function definition and recursive function calls
        return False
    if (name := name.parent()) and name.type == "function":
        return bool(name.name == function_name)
    return False


def belongs_to_class(name: Name, class_name: str) -> bool:
    """Check if given jedi Name is a direct child of the specified class."""
    while name := name.parent():
        if name.type == "class":
            return bool(name.name == class_name)
    return False


def belongs_to_function_qualified(name: Name, qualified_function_name: str) -> bool:
    """Check if the given jedi Name is a direct child of the specified function, matched by qualified function name."""
    try:
        if (
            name.full_name.startswith(name.module_name)
            and get_qualified_name(name.module_name, name.full_name) == qualified_function_name
        ):
            # Handles function definition and recursive function calls
            return False
        if (name := name.parent()) and name.type == "function":
            return get_qualified_name(name.module_name, name.full_name) == qualified_function_name
        return False
    except ValueError:
        return False
