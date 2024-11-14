import ast
from pathlib import Path

from codeflash.code_utils.static_analysis import (
    FunctionKind,
    ImportedInternalModuleAnalysis,
    analyze_imported_modules,
    function_kind,
    has_typed_parameters,
)
from codeflash.models.models import FunctionParent


def test_analyze_imported_modules() -> None:
    code_str = """
import os
import sys
import numpy as np
from . import mymodule
from datetime import datetime
from pandas import DataFrame
from pathlib import *
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeflash.code_utils.static_analysis import ImportedInternalModuleAnalysis

def a_function():
    from codeflash.code_utils.static_analysis import analyze_imported_modules
    from returns.result import Failure, Success
    pass
"""

    module_file_path = Path(__file__)
    project_root = (Path(__file__).parent.resolve() / "../").resolve()

    expected_imported_module_analysis = [
        ImportedInternalModuleAnalysis(
            name="static_analysis",
            full_name="codeflash.code_utils.static_analysis",
            file_path=project_root / Path("codeflash/code_utils/static_analysis.py"),
        ),
        ImportedInternalModuleAnalysis(
            name="mymodule", full_name="tests.mymodule", file_path=project_root / Path("tests/mymodule.py")
        ),
    ]
    actual_imported_module_analysis = analyze_imported_modules(code_str, module_file_path, project_root)
    assert set(actual_imported_module_analysis) == set(expected_imported_module_analysis)


def test_function_kind_typed() -> None:
    code1 = """
def a_function(a: int, b: str) -> None:
    pass
    """
    node1: ast.FunctionDef = ast.parse(code1).body[0]
    parents1: list[FunctionParent] = []
    assert function_kind(node1, parents1) == FunctionKind.FUNCTION
    assert has_typed_parameters(node1, parents1)
    code2 = """
def a_function(a: int, b) -> None:
    pass
        """
    node2: ast.FunctionDef = ast.parse(code2).body[0]
    parents2: list[FunctionParent] = []
    assert function_kind(node2, parents2) == FunctionKind.FUNCTION
    assert not has_typed_parameters(node2, parents2)
    code3 = """
def a_function() -> None:
    pass
"""
    node3: ast.FunctionDef = ast.parse(code3).body[0]
    parents3: list[FunctionParent] = []
    assert function_kind(node3, parents3) == FunctionKind.FUNCTION
    assert has_typed_parameters(node3, parents3)
    code4 = """
@staticmethod
def a_function(a, b) -> None:
    pass
"""
    node4: ast.FunctionDef = ast.parse(code4).body[0]
    parents4 = [FunctionParent(name="a_class", type="ClassDef")]
    assert function_kind(node4, parents4) == FunctionKind.STATIC_METHOD
    assert not has_typed_parameters(node4, parents4)
    code5 = """
@classmethod
def a_function(cls) -> None:
    pass
"""
    node5: ast.FunctionDef = ast.parse(code5).body[0]
    parents5 = [FunctionParent(name="a_class", type="ClassDef")]
    assert function_kind(node5, parents5) == FunctionKind.CLASS_METHOD
    assert has_typed_parameters(node5, parents5)
    code6 = """
@classmethod
def a_function(self, a) -> None:
    pass
"""
    node6: ast.FunctionDef = ast.parse(code6).body[0]
    parents6 = [FunctionParent(name="a_class", type="ClassDef")]
    assert function_kind(node6, parents6) == FunctionKind.CLASS_METHOD
    assert not has_typed_parameters(node6, parents6)
