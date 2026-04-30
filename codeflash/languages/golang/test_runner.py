from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
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

    others = _collect_other_test_files(test_file_paths)
    with _hide_other_test_files(others), _deduplicated_test_files(test_file_paths):
        run_regex = _build_run_regex(test_file_paths)
        cmd = ["go", "test", "-json", "-v", "-count=1"]
        if run_regex:
            cmd.extend(["-run", run_regex])
        cmd.extend(packages)
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

    test_file_paths = _collect_test_file_paths(test_paths, use_benchmarking=True)
    packages = _test_files_to_packages(test_file_paths, cwd)
    if not packages:
        packages = ["./..."]

    env = {**os.environ, **test_env}

    others = _collect_other_test_files(test_file_paths)
    with _hide_other_test_files(others), _deduplicated_test_files(test_file_paths):
        bench_regex = _build_bench_regex(test_file_paths)
        if bench_regex:
            benchtime_secs = min(target_duration_seconds, 1.0)
            num_benchmarks = len(_extract_func_names(test_file_paths, _BENCH_FUNC_RE))
            per_loop_estimate = int(num_benchmarks * benchtime_secs * 2) + 10
            cmd = [
                "go",
                "test",
                "-json",
                "-v",
                f"-bench={bench_regex}",
                f"-benchtime={benchtime_secs:.0f}s",
                # "-benchmem",
                "-count=1",  # setting count to as we looping manually to track timeout and max_loop
                "-run=^$",
                f"-timeout={per_loop_estimate}s",
                *packages,
            ]
            # logger.info("Benchmark command: %s", cmd)
            all_stdout: list[str] = []
            all_stderr: list[str] = []
            last_returncode = 0
            start_time = time.monotonic()
            for loop in range(1, max_loops + 1):
                proc_result = _run_cmd_kill_pg_on_timeout(cmd, cwd=cwd, env=env, timeout=per_loop_estimate)
                if proc_result.stdout:
                    all_stdout.append(proc_result.stdout)
                if proc_result.stderr:
                    all_stderr.append(proc_result.stderr)
                last_returncode = proc_result.returncode
                if proc_result.returncode != 0:
                    logger.warning(
                        "Benchmark loop %d failed (rc=%d):\nstdout:%s\nstderr: %s",
                        loop,
                        proc_result.returncode,
                        proc_result.stdout,
                        proc_result.stderr,
                    )
                    break
                elapsed = time.monotonic() - start_time
                if loop >= min_loops and elapsed >= target_duration_seconds:
                    logger.info(
                        "Benchmark stopping after %d loops (%.1fs elapsed, target %.1fs)",
                        loop,
                        elapsed,
                        target_duration_seconds,
                    )
                    break
            logger.info("Benchmark completed %d loop(s), returncode: %d", loop, last_returncode)
            combined_stdout = "".join(all_stdout)
            combined_stderr = "".join(all_stderr)
            proc_result = subprocess.CompletedProcess(
                args=cmd, returncode=last_returncode, stdout=combined_stdout, stderr=combined_stderr
            )
        else:
            logger.warning("No Benchmark* functions found in perf test files: %s", [str(p) for p in test_file_paths])
            proc_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

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


def _collect_test_file_paths(test_paths: Any, *, use_benchmarking: bool = False) -> list[Path]:
    from pathlib import Path as _Path

    if test_paths is None:
        return []

    if hasattr(test_paths, "test_files"):
        paths = []
        for tf in test_paths.test_files:
            if use_benchmarking:
                p = getattr(tf, "benchmarking_file_path", None) or getattr(tf, "perf_file_path", None)
            else:
                p = getattr(tf, "instrumented_behavior_file_path", None)
            if p is None:
                p = getattr(tf, "original_file_path", None)
            if p is not None:
                paths.append(_Path(p))
        return paths

    if isinstance(test_paths, list):
        return [_Path(p) for p in test_paths]

    return []


def _collect_other_test_files(test_file_paths: list[Path]) -> list[Path]:

    if not test_file_paths:
        return []

    keep = {f.resolve() for f in test_file_paths}
    dirs = {f.resolve().parent for f in test_file_paths}

    others: list[Path] = []
    for d in dirs:
        for f in d.glob("*_test.go"):
            if f.resolve() not in keep and f.exists():
                others.append(f)
    return others


@contextlib.contextmanager
def _hide_other_test_files(others: list[Path]) -> Generator[None, None, None]:
    """Temporarily rename test files we don't want compiled.

    Go compiles ALL *_test.go files in a package together, so any duplicate
    symbols across test files cause build errors. We hide every test file in
    the target directories except the ones we intend to run.
    """
    renamed: list[tuple[Path, Path]] = []
    for f in others:
        hidden = f.with_suffix(".go.codeflash_hidden")
        try:
            f.rename(hidden)
            renamed.append((hidden, f))
            logger.debug("Temporarily hid %s during go test", f)
        except OSError:
            logger.debug("Could not hide %s, skipping", f)
    try:
        yield
    finally:
        for hidden, original in renamed:
            try:
                hidden.rename(original)
                logger.debug("Restored %s", original)
            except OSError:
                logger.warning("Failed to restore %s from %s", original, hidden)


_TEST_FUNC_RE = re.compile(r"^func\s+(Test\w+)\s*\(", re.MULTILINE)
_BENCH_FUNC_RE = re.compile(r"^func\s+(Benchmark\w+)\s*\(", re.MULTILINE)
_FUNC_DECL_RE = re.compile(r"^(func\s+)(Test\w+|Benchmark\w+)(\s*\()", re.MULTILINE)


def _extract_func_names(test_files: list[Path], pattern: re.Pattern[str]) -> list[str]:
    names: list[str] = []
    for f in test_files:
        try:
            content = f.read_text(encoding="utf-8")
        except OSError:
            continue
        names.extend(pattern.findall(content))
    return names


def _build_run_regex(test_files: list[Path]) -> str | None:
    names = _extract_func_names(test_files, _TEST_FUNC_RE)
    if not names:
        return None
    return f"^({'|'.join(re.escape(n) for n in names)})$"


def _build_bench_regex(test_files: list[Path]) -> str | None:
    names = _extract_func_names(test_files, _BENCH_FUNC_RE)
    if not names:
        return None
    return f"^({'|'.join(re.escape(n) for n in names)})$"


def _deduplicate_test_func_names(test_files: list[Path]) -> dict[Path, str]:
    seen: dict[str, int] = {}
    originals: dict[Path, str] = {}

    for f in test_files:
        try:
            content = f.read_text(encoding="utf-8")
        except OSError:
            continue

        names_in_file = [name for _, name, _ in _FUNC_DECL_RE.findall(content)]
        if not names_in_file:
            continue

        needs_rewrite = any(name in seen for name in names_in_file)

        if not needs_rewrite:
            for name in names_in_file:
                seen[name] = 1
            continue

        originals[f] = content

        def _renamer(m: re.Match[str]) -> str:
            prefix, name, suffix = m.group(1), m.group(2), m.group(3)
            if name not in seen:
                seen[name] = 1
                return m.group(0)
            idx = seen[name]
            seen[name] = idx + 1
            return f"{prefix}{name}_{idx}{suffix}"

        new_content = _FUNC_DECL_RE.sub(_renamer, content)
        f.write_text(new_content, encoding="utf-8")
        logger.debug("Deduplicated test function names in %s", f)

    return originals


@contextlib.contextmanager
def _deduplicated_test_files(test_files: list[Path]) -> Generator[None, None, None]:
    originals = _deduplicate_test_func_names(test_files)
    try:
        yield
    finally:
        for f, content in originals.items():
            try:
                f.write_text(content, encoding="utf-8")
            except OSError:
                logger.warning("Failed to restore original content for %s", f)


def _test_files_to_packages(test_files: list[Path], cwd: Path) -> list[str]:
    dirs: set[str] = set()
    resolved_cwd = cwd.resolve()
    for f in test_files:
        try:
            rel = f.resolve().parent.relative_to(resolved_cwd)
            pkg = f"./{rel.as_posix()}" if rel.parts else "."
            dirs.add(pkg)
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
