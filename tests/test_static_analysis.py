from pathlib import Path

from codeflash.code_utils.static_analysis import ImportedInternalModuleAnalysis, analyze_imported_internal_modules


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
    actual_imported_module_analysis = analyze_imported_internal_modules(code_str, module_file_path, project_root)
    assert set(actual_imported_module_analysis) == set(expected_imported_module_analysis)
