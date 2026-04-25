from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
from typing import TYPE_CHECKING, Any

from codeflash.languages.base import TestResult

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

logger = logging.getLogger(__name__)


def run_behavioral_tests(
    test_paths: Any,
    test_env: dict[str, str],
    cwd: Path,
    timeout: int | None = None,
    project_root: Path | None = None,
    enable_coverage: bool = False,
    candidate_index: int = 0,
) -> tuple[Path, subprocess.CompletedProcess[str], Path | None, Path | None]:
    result_dir = cwd / ".codeflash" / "go_test_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    json_output_file = result_dir / f"behavioral_{candidate_index}.jsonl"

    test_file_paths = _collect_test_file_paths(test_paths)
    packages = _test_files_to_packages(test_file_paths, cwd)
    if not packages:
        packages = ["./..."]

    env = {**os.environ, **test_env}

    cmd = ["go", "test", "-json", "-v", "-count=1", *packages]

    originals = _collect_original_file_paths(test_paths)
    with _hide_original_test_files(originals):
        proc_result = _run_cmd_kill_pg_on_timeout(cmd, cwd=cwd, env=env, timeout=timeout)

    json_output_file.write_text(proc_result.stdout or "", encoding="utf-8")

    return json_output_file, proc_result, None, None


def run_benchmarking_tests(
    test_paths: Any,
    test_env: dict[str, str],
    cwd: Path,
    timeout: int | None = None,
    project_root: Path | None = None,
    min_loops: int = 5,
    max_loops: int = 100_000,
    target_duration_seconds: float = 10.0,
    inner_iterations: int = 100,
) -> tuple[Path, subprocess.CompletedProcess[str]]:
    result_dir = cwd / ".codeflash" / "go_test_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    json_output_file = result_dir / "benchmark.jsonl"

    test_file_paths = _collect_test_file_paths(test_paths)
    packages = _test_files_to_packages(test_file_paths, cwd)
    if not packages:
        packages = ["./..."]

    env = {**os.environ, **test_env}

    benchtime = f"{target_duration_seconds:.0f}s"
    cmd = [
        "go",
        "test",
        "-json",
        "-v",
        "-bench=.",
        f"-benchtime={benchtime}",
        "-benchmem",
        f"-count={min_loops}",
        "-run=^$",
        *packages,
    ]

    originals = _collect_original_file_paths(test_paths)
    with _hide_original_test_files(originals):
        proc_result = _run_cmd_kill_pg_on_timeout(cmd, cwd=cwd, env=env, timeout=timeout)

    json_output_file.write_text(proc_result.stdout or "", encoding="utf-8")

    return json_output_file, proc_result


def parse_go_test_json(json_output: str) -> list[TestResult]:
    results: dict[str, TestResult] = {}

    for line in json_output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        action = event.get("Action")
        test_name = event.get("Test")
        if test_name is None:
            continue

        package = event.get("Package", "")

        if action == "pass":
            elapsed = event.get("Elapsed", 0)
            results[test_name] = TestResult(
                test_name=test_name,
                test_file=_package_to_path(package),
                passed=True,
                runtime_ns=int(elapsed * 1_000_000_000) if elapsed else None,
            )
        elif action == "fail":
            elapsed = event.get("Elapsed", 0)
            existing = results.get(test_name)
            stdout = existing.stdout if existing else ""
            results[test_name] = TestResult(
                test_name=test_name,
                test_file=_package_to_path(package),
                passed=False,
                runtime_ns=int(elapsed * 1_000_000_000) if elapsed else None,
                stdout=stdout,
                error_message=f"Test {test_name} failed",
            )
        elif action == "output":
            output_text = event.get("Output", "")
            if test_name in results:
                results[test_name] = TestResult(
                    test_name=results[test_name].test_name,
                    test_file=results[test_name].test_file,
                    passed=results[test_name].passed,
                    runtime_ns=results[test_name].runtime_ns,
                    stdout=results[test_name].stdout + output_text,
                    stderr=results[test_name].stderr,
                    error_message=results[test_name].error_message,
                )
            else:
                results[test_name] = TestResult(
                    test_name=test_name, test_file=_package_to_path(package), passed=True, stdout=output_text
                )

    return list(results.values())


def parse_test_results(json_output_path: Path, stdout: str) -> list[TestResult]:
    try:
        content = json_output_path.read_text(encoding="utf-8")
    except Exception:
        content = stdout
    return parse_go_test_json(content)


def _package_to_path(package: str) -> Path:
    from pathlib import Path as _Path

    if package:
        return _Path(package.replace("/", os.sep))
    return _Path()


def _collect_test_file_paths(test_paths: Any) -> list[Path]:
    from pathlib import Path as _Path

    if test_paths is None:
        return []

    if hasattr(test_paths, "test_files"):
        paths = []
        for tf in test_paths.test_files:
            p = getattr(tf, "instrumented_behavior_file_path", None) or getattr(tf, "original_file_path", None)
            if p is not None:
                paths.append(_Path(p))
        return paths

    if isinstance(test_paths, list):
        return [_Path(p) for p in test_paths]

    return []


def _collect_original_file_paths(test_paths: Any) -> list[Path]:
    from pathlib import Path as _Path

    if test_paths is None or not hasattr(test_paths, "test_files"):
        return []

    originals: list[Path] = []
    for tf in test_paths.test_files:
        instrumented = getattr(tf, "instrumented_behavior_file_path", None)
        original = getattr(tf, "original_file_path", None)
        if instrumented is not None and original is not None:
            instrumented_p = _Path(instrumented)
            original_p = _Path(original)
            if instrumented_p != original_p and original_p.exists():
                originals.append(original_p)
    return originals


@contextlib.contextmanager
def _hide_original_test_files(originals: list[Path]) -> Generator[None, None, None]:
    """Temporarily rename original test files so `go test` only sees the instrumented copies.

    Go compiles all *_test.go files in a package together, so having both the original
    and its instrumented copy causes duplicate symbol errors.
    """
    renamed: list[tuple[Path, Path]] = []
    for original in originals:
        hidden = original.with_suffix(".go.codeflash_hidden")
        try:
            original.rename(hidden)
            renamed.append((hidden, original))
            logger.debug("Temporarily hid %s during go test", original)
        except OSError:
            logger.debug("Could not hide %s, skipping", original)
    try:
        yield
    finally:
        for hidden, original in renamed:
            try:
                hidden.rename(original)
                logger.debug("Restored %s", original)
            except OSError:
                logger.warning("Failed to restore %s from %s", original, hidden)


def _test_files_to_packages(test_files: list[Path], cwd: Path) -> list[str]:
    dirs: set[str] = set()
    for f in test_files:
        try:
            rel = f.resolve().parent.relative_to(cwd.resolve())
            dirs.add(f"./{rel.as_posix()}")
        except ValueError:
            continue
    return sorted(dirs) if dirs else []


def _run_cmd_kill_pg_on_timeout(
    cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, timeout: int | None = None
) -> subprocess.CompletedProcess[str]:
    if sys.platform == "win32":
        try:
            return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(
                args=cmd, returncode=-2, stdout="", stderr=f"Process timed out after {timeout}s"
            )

    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(args=cmd, returncode=proc.returncode, stdout=stdout, stderr=stderr)
    except subprocess.TimeoutExpired:
        pgid = None
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if pgid is not None:
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
            proc.wait()
        try:
            stdout_data = proc.stdout.read() if proc.stdout else ""
            stderr_data = proc.stderr.read() if proc.stderr else ""
        except Exception:
            stdout_data, stderr_data = "", ""
        return subprocess.CompletedProcess(
            args=cmd, returncode=-2, stdout=stdout_data, stderr=stderr_data or f"Process timed out after {timeout}s"
        )
