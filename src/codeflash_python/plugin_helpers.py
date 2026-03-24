"""Standalone helper functions used by PythonPlugin methods."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from codeflash_core.models import CoverageData, FunctionToOptimize
    from codeflash_python.models.models import OptimizedCandidateSource

logger = logging.getLogger(__name__)


def make_test_env(
    project_root: Path | str, *, loop_index: int = 0, test_iteration: int = 0, tracer_disable: int = 1
) -> dict[str, str]:
    """Return a copy of os.environ configured for running codeflash tests.

    Matches original codeflash get_test_env(): prepends project_root to PYTHONPATH
    and sets CODEFLASH_* env vars expected by instrumented test harness.
    """
    env = os.environ.copy()
    project_root_str = str(project_root)
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{project_root_str}{os.pathsep}{pythonpath}"
    else:
        env["PYTHONPATH"] = project_root_str
    env["CODEFLASH_LOOP_INDEX"] = str(loop_index)
    env["CODEFLASH_TEST_ITERATION"] = str(test_iteration)
    env["CODEFLASH_TRACER_DISABLE"] = str(tracer_disable)
    return env


def format_speedup_pct(speedup: float) -> str:
    """Format speedup as percentage string matching original codeflash API format."""
    return f"{int(speedup * 100)}%"


def read_return_values(test_iteration: int) -> dict[str, list[object]]:
    """Read return values from the SQLite file written by instrumented tests.

    Returns a dict mapping test_function_name -> list of deserialized return values.
    Only reads rows with loop_index == 1 (first timing iteration), matching original behavior.
    """
    import pickle
    import sqlite3

    from codeflash_python.code_utils.code_utils import get_run_tmp_file

    sqlite_path = get_run_tmp_file(Path(f"test_return_values_{test_iteration}.sqlite"))
    if not sqlite_path.exists():
        return {}

    result: dict[str, list[object]] = {}
    db = None
    try:
        db = sqlite3.connect(sqlite_path)
        rows = db.execute("SELECT test_function_name, loop_index, return_value FROM test_results").fetchall()
        db.close()
        db = None

        for test_fn_name, loop_index, return_value_blob in rows:
            if loop_index != 1 or not return_value_blob or not test_fn_name:
                continue
            try:
                ret_val = pickle.loads(return_value_blob)
                result.setdefault(test_fn_name, []).append(ret_val)
            except Exception:
                logger.debug("Failed to deserialize return value for %s", test_fn_name)
    except Exception:
        logger.debug("Failed to read return values from %s", sqlite_path)
    finally:
        if db is not None:
            db.close()

    return result


def map_candidate_source(source: str) -> OptimizedCandidateSource:
    """Map core Candidate.source string to OptimizedCandidateSource enum value."""
    from codeflash_python.models.models import OptimizedCandidateSource

    mapping = {
        "optimize": OptimizedCandidateSource.OPTIMIZE,
        "line_profiler": OptimizedCandidateSource.OPTIMIZE_LP,
        "refine": OptimizedCandidateSource.REFINE,
        "repair": OptimizedCandidateSource.REPAIR,
        "adaptive": OptimizedCandidateSource.ADAPTIVE,
    }
    return mapping.get(source, OptimizedCandidateSource.OPTIMIZE)


def coverage_data_to_details_dict(cov_data: CoverageData) -> dict[str, Any]:
    """Convert CoverageData to the dict format expected by the repair API."""
    mc = cov_data.main_func_coverage
    details: dict[str, Any] = {
        "coverage_percentage": cov_data.coverage,
        "threshold_percentage": cov_data.threshold_percentage,
        "main_function": {
            "name": mc.name,
            "coverage": mc.coverage,
            "executed_lines": sorted(mc.executed_lines),
            "unexecuted_lines": sorted(mc.unexecuted_lines),
            "executed_branches": mc.executed_branches,
            "unexecuted_branches": mc.unexecuted_branches,
        },
    }
    dc = cov_data.dependent_func_coverage
    if dc:
        details["dependent_function"] = {
            "name": dc.name,
            "coverage": dc.coverage,
            "executed_lines": sorted(dc.executed_lines),
            "unexecuted_lines": sorted(dc.unexecuted_lines),
            "executed_branches": dc.executed_branches,
            "unexecuted_branches": dc.unexecuted_branches,
        }
    return details


def replace_function_simple(source: str, function: FunctionToOptimize, new_source: str) -> str:
    from codeflash_python.static_analysis.code_replacer import replace_functions_in_file

    try:
        return replace_functions_in_file(
            source_code=source,
            original_function_names=[function.qualified_name],
            optimized_code=new_source,
            preexisting_objects=set(),
        )
    except Exception:
        logger.warning("Failed to replace function %s", function.function_name)
        return source


def format_code_with_ruff_or_black(source: str, file_path: Path | None = None) -> str:
    import subprocess

    try:
        result = subprocess.run(
            ["ruff", "format", "-"], check=False, input=source, capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["black", "-q", "-"], check=False, input=source, capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    except Exception:
        pass

    return source
