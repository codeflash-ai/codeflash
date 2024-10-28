from pathlib import Path

from codeflash.code_utils.code_extractor import add_needed_imports_from_module


def test_add_needed_imports_from_module0() -> None:
    src_module = '''import ast
import logging
import os
from typing import Union
import jedi
import tiktoken
from jedi.api.classes import Name
from pydantic.dataclasses import dataclass
from codeflash.code_utils.code_extractor import get_code, get_code_no_skeleton
from codeflash.code_utils.code_utils import path_belongs_to_site_packages
from codeflash.discovery.functions_to_optimize import FunctionParent, FunctionToOptimize

def belongs_to_class(name: Name, class_name: str) -> bool:
    """Check if the given name belongs to the specified class."""
    if name.full_name and name.full_name.startswith(f"{name.module_name}.{class_name}."):
        return True
    return False

def heyjude() -> None:
    print("Hey Jude, don't make it bad")

def belongs_to_function(name: Name, function_name: str) -> bool:
    """Check if the given name belongs to the specified function"""
    if name.full_name and name.full_name.startswith(name.module_name):
        subname: str = name.full_name.replace(name.module_name, "", 1)
    else:
        return False
    # The name is defined inside the function or is the function itself
    return f".{function_name}." in subname or f".{function_name}" == subname

@dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class Source:
    full_name: str
    definition: Name
    source_code: str
'''

    dst_module = """def heyjude() -> None:
    print("Hey Jude, don't make it bad")
"""

    expected = """def heyjude() -> None:
    print("Hey Jude, don't make it bad")
"""
    src_path = Path("/home/roger/repos/codeflash/cli/codeflash/optimization/function_context.py")
    dst_path = Path("/home/roger/repos/codeflash/cli/codeflash/optimization/function_context.py")
    project_root = Path("/home/roger/repos/codeflash")
    new_module = add_needed_imports_from_module(src_module, dst_module, src_path, dst_path, project_root)
    assert new_module == expected


def test_add_needed_imports_from_module() -> None:
    src_module = '''import ast
import logging
import os
from typing import Union

import jedi
import tiktoken
from jedi.api.classes import Name
from pydantic.dataclasses import dataclass

from codeflash.code_utils.code_extractor import get_code, get_code_no_skeleton
from codeflash.code_utils.code_utils import path_belongs_to_site_packages
from codeflash.discovery.functions_to_optimize import FunctionParent, FunctionToOptimize


def belongs_to_class(name: Name, class_name: str) -> bool:
    """Check if the given name belongs to the specified class."""
    if name.full_name and name.full_name.startswith(f"{name.module_name}.{class_name}."):
        return True
    return False


def belongs_to_function(name: Name, function_name: str) -> bool:
    """Check if the given name belongs to the specified function"""
    if name.full_name and name.full_name.startswith(name.module_name):
        subname: str = name.full_name.replace(name.module_name, "", 1)
    else:
        return False
    # The name is defined inside the function or is the function itself
    return f".{function_name}." in subname or f".{function_name}" == subname


@dataclass(frozen=True, config={"arbitrary_types_allowed": True})
class Source:
    full_name: str
    definition: Name
    source_code: str
'''

    dst_module = '''def belongs_to_function(name: Name, function_name: str) -> bool:
    """Check if the given name belongs to the specified function"""
    if name.full_name and name.full_name.startswith(name.module_name):
        subname: str = name.full_name.replace(name.module_name, "", 1)
    else:
        return False
    # The name is defined inside the function or is the function itself
    return f".{function_name}." in subname or f".{function_name}" == subname
'''

    expected = '''from jedi.api.classes import Name

def belongs_to_function(name: Name, function_name: str) -> bool:
    """Check if the given name belongs to the specified function"""
    if name.full_name and name.full_name.startswith(name.module_name):
        subname: str = name.full_name.replace(name.module_name, "", 1)
    else:
        return False
    # The name is defined inside the function or is the function itself
    return f".{function_name}." in subname or f".{function_name}" == subname
'''
    src_path = Path("/home/roger/repos/codeflash/cli/codeflash/optimization/function_context.py")
    dst_path = Path("/home/roger/repos/codeflash/cli/codeflash/optimization/function_context.py")
    project_root = Path("/home/roger/repos/codeflash")
    new_module = add_needed_imports_from_module(src_module, dst_module, src_path, dst_path, project_root)
    assert new_module == expected
