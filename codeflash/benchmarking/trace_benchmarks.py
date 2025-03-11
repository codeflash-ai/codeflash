from __future__ import annotations
from codeflash.code_utils.compat import SAFE_SYS_EXECUTABLE
from pathlib import Path
import subprocess

def trace_benchmarks_pytest(benchmarks_root: Path, project_root: Path) -> None:
    result = subprocess.run(
        [
            SAFE_SYS_EXECUTABLE,
            Path(__file__).parent / "pytest_new_process_trace_benchmarks.py",
            benchmarks_root,
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)
