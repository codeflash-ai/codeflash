from __future__ import annotations

import json
import os
import pstats
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from codeflash.cli_cmds.console import logger


def run_cprofile_stage(test_root: Path, project_root: Path, output_trace_file: Path) -> bool:
    with tempfile.NamedTemporaryFile(suffix=".prof", delete=False) as tmp:
        prof_file = Path(tmp.name)

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)

        cmd = [
            sys.executable,
            "-m",
            "cProfile",
            "-o",
            str(prof_file),
            "-m",
            "pytest",
            str(test_root),
            "-x",
            "-q",
            "--tb=short",
            "-p",
            "no:warnings",
        ]

        logger.debug(f"Running cProfile stage: {' '.join(cmd)}")

        result = subprocess.run(cmd, check=False, env=env, cwd=str(project_root))

        if result.returncode not in {0, 1}:
            logger.warning(f"cProfile stage returned {result.returncode}")

        if not prof_file.exists() or prof_file.stat().st_size == 0:
            logger.warning("cProfile output file is empty or missing")
            return False

        stats = pstats.Stats(str(prof_file))
        convert_pstats_to_sqlite(stats, output_trace_file, project_root)

        logger.info(f"cProfile stage complete: {len(stats.stats)} functions profiled")  # type: ignore[attr-defined]

    except Exception as e:
        logger.warning(f"cProfile stage failed: {e}")
        return False
    else:
        return True
    finally:
        if prof_file.exists():
            prof_file.unlink()


def convert_pstats_to_sqlite(stats: pstats.Stats, output_file: Path, project_root: Path) -> None:
    if output_file.exists():
        output_file.unlink()

    con = sqlite3.connect(output_file)
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE pstats (
            filename TEXT,
            line_number INTEGER,
            function TEXT,
            class_name TEXT,
            call_count_nonrecursive INTEGER,
            num_callers INTEGER,
            total_time_ns INTEGER,
            cumulative_time_ns INTEGER,
            callers TEXT
        )
    """)
    cur.execute("CREATE TABLE total_time (time_ns INTEGER)")

    total_time_s = 0.0
    project_root_str = str(project_root.resolve())

    for func_key, (cc, nc, tt, ct, callers) in stats.stats.items():  # type: ignore[attr-defined]
        filename, line_number, func_name = func_key

        if not _is_project_file(filename, project_root_str):
            continue

        total_time_s += tt

        class_name = None
        base_func_name = func_name
        if "." in func_name and not func_name.startswith("<"):
            parts = func_name.rsplit(".", 1)
            if len(parts) == 2:
                class_name, base_func_name = parts

        callers_json = json.dumps([{"key": list(k), "value": v} for k, v in callers.items()])
        total_time_ns = int(tt * 1e9)
        cumulative_time_ns = int(ct * 1e9)

        cur.execute(
            "INSERT INTO pstats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                filename,
                line_number,
                base_func_name,
                class_name,
                cc,
                nc,
                total_time_ns,
                cumulative_time_ns,
                callers_json,
            ),
        )

    total_time_ns = int(total_time_s * 1e9)
    cur.execute("INSERT INTO total_time VALUES (?)", (total_time_ns,))

    con.commit()
    con.close()


def _is_project_file(filename: str, project_root: str) -> bool:
    if not filename or filename.startswith("<"):
        return False

    try:
        abs_filename = str(Path(filename).resolve())
    except (OSError, ValueError):
        return False

    if not abs_filename.startswith(project_root):
        return False

    exclude_patterns = ("site-packages", ".venv", "venv", "__pycache__", ".pyc", "_pytest", "pluggy")
    return not any(pattern in abs_filename for pattern in exclude_patterns)
