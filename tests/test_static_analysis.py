from pathlib import Path

from codeflash.code_utils.static_analysis import ImportedModuleAnalysis, analyze_imported_modules


def test_analyze_imported_modules() -> None:
    code_str = """
import os
import sys
import numpy as np
from . import mymodule
from datetime import datetime
from pandas import DataFrame
from pathlib import *
from codeflash.code_utils.static_analysis import analyze_imported_modules
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace

def afunction():
    import datetime
    from returns.result import Failure, Success
    pass
"""

    module_file_path = Path(__file__)
    project_root = (Path(__file__).parent.resolve() / "../").resolve()

    expected_imported_module_analysis = [
        ImportedModuleAnalysis(
            name="result",
            origin="standard library",
            full_name="returns.result",
            file_path=Path("/Users/renaud/miniforge3/envs/codeflash312/lib/python3.12/site-packages/returns/result.py"),
        ),
        ImportedModuleAnalysis(
            name="pandas",
            origin="standard library",
            full_name="pandas",
            file_path=Path(
                "/Users/renaud/miniforge3/envs/codeflash312/lib/python3.12/site-packages/pandas/__init__.py"
            ),
        ),
        ImportedModuleAnalysis(name="sys", origin="standard library", full_name="sys", file_path=None),
        ImportedModuleAnalysis(
            name="typing",
            origin="standard library",
            full_name="typing",
            file_path=Path("/Users/renaud/miniforge3/envs/codeflash312/lib/python3.12/typing.py"),
        ),
        ImportedModuleAnalysis(
            name="numpy",
            origin="standard library",
            full_name="numpy",
            file_path=Path("/Users/renaud/miniforge3/envs/codeflash312/lib/python3.12/site-packages/numpy/__init__.py"),
        ),
        ImportedModuleAnalysis(
            name="datetime",
            origin="standard library",
            full_name="datetime",
            file_path=Path("/Users/renaud/miniforge3/envs/codeflash312/lib/python3.12/datetime.py"),
        ),
        ImportedModuleAnalysis(
            name="argparse",
            origin="standard library",
            full_name="argparse",
            file_path=Path("/Users/renaud/miniforge3/envs/codeflash312/lib/python3.12/argparse.py"),
        ),
        ImportedModuleAnalysis(name="os", origin="standard library", full_name="os", file_path=None),
        ImportedModuleAnalysis(
            name="static_analysis",
            origin="internal",
            full_name="codeflash.code_utils.static_analysis",
            file_path=Path("/Users/renaud/repos/codeflash/cli/codeflash/code_utils/static_analysis.py"),
        ),
        ImportedModuleAnalysis(
            name="mymodule",
            origin="internal",
            full_name="tests.mymodule",
            file_path=Path("/Users/renaud/repos/codeflash/cli/tests/mymodule.py"),
        ),
        ImportedModuleAnalysis(
            name="pathlib",
            origin="standard library",
            full_name="pathlib",
            file_path=Path("/Users/renaud/miniforge3/envs/codeflash312/lib/python3.12/pathlib.py"),
        ),
    ]
    actual_imported_module_analysis = analyze_imported_modules(code_str, module_file_path, project_root)
    assert set(actual_imported_module_analysis) == set(expected_imported_module_analysis)
