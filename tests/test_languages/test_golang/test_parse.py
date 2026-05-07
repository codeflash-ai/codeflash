from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from codeflash.languages.golang.parse import BENCHMARK_RE, parse_go_test_output
from codeflash.models.models import TestFile, TestFiles
from codeflash.models.test_type import TestType


def _make_test_config(tmp_path: Path) -> MagicMock:
    cfg = MagicMock()
    cfg.tests_project_rootdir = tmp_path
    cfg.test_framework = "go-test"
    return cfg


def _make_test_files(tmp_path: Path, filenames: list[str] | None = None, test_type: TestType = TestType.GENERATED_REGRESSION) -> TestFiles:
    if filenames is None:
        filenames = ["calc_test.go"]
    files = []
    for name in filenames:
        path = (tmp_path / name).resolve()
        path.write_text("package calc\n", encoding="utf-8")
        files.append(
            TestFile(
                instrumented_behavior_file_path=path,
                test_type=test_type,
            )
        )
    return TestFiles(test_files=files)


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")


class TestBenchmarkRegex:
    def test_basic_benchmark_line(self) -> None:
        line = "BenchmarkAdd-8   \t 1000000\t      1234 ns/op\t      56 B/op\t       2 allocs/op"
        m = BENCHMARK_RE.search(line)
        assert m is not None
        assert m.group(1) == "BenchmarkAdd"
        assert m.group(2) == "1000000"
        assert m.group(3) == "1234"
        assert m.group(4) == "56"
        assert m.group(5) == "2"

    def test_benchmark_without_mem(self) -> None:
        line = "BenchmarkSort   5000   300000 ns/op"
        m = BENCHMARK_RE.search(line)
        assert m is not None
        assert m.group(1) == "BenchmarkSort"
        assert m.group(4) is None
        assert m.group(5) is None

    def test_benchmark_with_float_ns(self) -> None:
        line = "BenchmarkFib-16   100000   12345.67 ns/op"
        m = BENCHMARK_RE.search(line)
        assert m is not None
        assert m.group(3) == "12345.67"

    def test_non_benchmark_line(self) -> None:
        line = "=== RUN   TestAdd"
        m = BENCHMARK_RE.search(line)
        assert m is None


class TestParseGoTestOutputBehavioral:
    def test_all_passing(self, tmp_path: Path) -> None:
        events = [
            {"Time": "2024-01-01T00:00:00Z", "Action": "run", "Package": "example/calc", "Test": "TestAdd"},
            {"Time": "2024-01-01T00:00:00Z", "Action": "output", "Package": "example/calc", "Test": "TestAdd", "Output": "=== RUN   TestAdd\n"},
            {"Time": "2024-01-01T00:00:00Z", "Action": "output", "Package": "example/calc", "Test": "TestAdd", "Output": "--- PASS: TestAdd (0.00s)\n"},
            {"Time": "2024-01-01T00:00:00Z", "Action": "pass", "Package": "example/calc", "Test": "TestAdd", "Elapsed": 0.001},
            {"Time": "2024-01-01T00:00:00Z", "Action": "run", "Package": "example/calc", "Test": "TestSub"},
            {"Time": "2024-01-01T00:00:00Z", "Action": "pass", "Package": "example/calc", "Test": "TestSub", "Elapsed": 0.002},
            {"Time": "2024-01-01T00:00:00Z", "Action": "pass", "Package": "example/calc"},
        ]

        json_path = (tmp_path / "results.jsonl").resolve()
        _write_jsonl(json_path, events)

        test_files = _make_test_files(tmp_path)
        cfg = _make_test_config(tmp_path)

        results = parse_go_test_output(json_path, test_files, cfg)
        assert len(results.test_results) == 2

        by_name = {r.id.test_function_name: r for r in results.test_results}
        assert by_name["TestAdd"].did_pass is True
        assert by_name["TestAdd"].runtime == 1_000_000
        assert by_name["TestSub"].did_pass is True
        assert by_name["TestSub"].runtime == 2_000_000

    def test_with_failure(self, tmp_path: Path) -> None:
        events = [
            {"Action": "run", "Package": "example/calc", "Test": "TestAdd"},
            {"Action": "output", "Package": "example/calc", "Test": "TestAdd", "Output": "got 4, want 5\n"},
            {"Action": "fail", "Package": "example/calc", "Test": "TestAdd", "Elapsed": 0.01},
        ]

        json_path = (tmp_path / "results.jsonl").resolve()
        _write_jsonl(json_path, events)

        test_files = _make_test_files(tmp_path)
        cfg = _make_test_config(tmp_path)

        results = parse_go_test_output(json_path, test_files, cfg)
        assert len(results.test_results) == 1
        assert results.test_results[0].did_pass is False
        assert "got 4, want 5" in results.test_results[0].stdout

    def test_mixed_pass_fail(self, tmp_path: Path) -> None:
        events = [
            {"Action": "pass", "Package": "p", "Test": "TestA", "Elapsed": 0.001},
            {"Action": "fail", "Package": "p", "Test": "TestB", "Elapsed": 0.002},
            {"Action": "pass", "Package": "p", "Test": "TestC", "Elapsed": 0.003},
        ]

        json_path = (tmp_path / "results.jsonl").resolve()
        _write_jsonl(json_path, events)

        test_files = _make_test_files(tmp_path)
        cfg = _make_test_config(tmp_path)

        results = parse_go_test_output(json_path, test_files, cfg)
        by_name = {r.id.test_function_name: r for r in results.test_results}
        assert by_name["TestA"].did_pass is True
        assert by_name["TestB"].did_pass is False
        assert by_name["TestC"].did_pass is True

    def test_empty_file(self, tmp_path: Path) -> None:
        json_path = (tmp_path / "empty.jsonl").resolve()
        json_path.write_text("", encoding="utf-8")

        test_files = _make_test_files(tmp_path)
        cfg = _make_test_config(tmp_path)

        results = parse_go_test_output(json_path, test_files, cfg)
        assert len(results.test_results) == 0

    def test_missing_file_falls_back_to_run_result(self, tmp_path: Path) -> None:
        json_path = (tmp_path / "nonexistent.jsonl").resolve()
        events = [
            {"Action": "pass", "Package": "p", "Test": "TestX", "Elapsed": 0.005},
        ]
        stdout = "\n".join(json.dumps(e) for e in events)
        run_result = subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")

        test_files = _make_test_files(tmp_path)
        cfg = _make_test_config(tmp_path)

        results = parse_go_test_output(json_path, test_files, cfg, run_result)
        assert len(results.test_results) == 1
        assert results.test_results[0].id.test_function_name == "TestX"

    def test_invalid_json_lines_skipped(self, tmp_path: Path) -> None:
        content = 'not json\n{"Action":"pass","Package":"p","Test":"TestOK","Elapsed":0.001}\nalso bad\n'
        json_path = (tmp_path / "results.jsonl").resolve()
        json_path.write_text(content, encoding="utf-8")

        test_files = _make_test_files(tmp_path)
        cfg = _make_test_config(tmp_path)

        results = parse_go_test_output(json_path, test_files, cfg)
        assert len(results.test_results) == 1
        assert results.test_results[0].did_pass is True

    def test_test_type_from_test_files(self, tmp_path: Path) -> None:
        test_files = _make_test_files(tmp_path, test_type=TestType.EXISTING_UNIT_TEST)
        events = [
            {"Action": "pass", "Package": "p", "Test": "TestFoo", "Elapsed": 0.001},
        ]
        json_path = (tmp_path / "results.jsonl").resolve()
        _write_jsonl(json_path, events)
        cfg = _make_test_config(tmp_path)

        results = parse_go_test_output(json_path, test_files, cfg)
        assert results.test_results[0].test_type == TestType.EXISTING_UNIT_TEST

    def test_framework_is_go_test(self, tmp_path: Path) -> None:
        events = [
            {"Action": "pass", "Package": "p", "Test": "TestBar", "Elapsed": 0.001},
        ]
        json_path = (tmp_path / "results.jsonl").resolve()
        _write_jsonl(json_path, events)
        test_files = _make_test_files(tmp_path)
        cfg = _make_test_config(tmp_path)

        results = parse_go_test_output(json_path, test_files, cfg)
        assert results.test_results[0].test_framework == "go-test"

    def test_package_level_events_ignored(self, tmp_path: Path) -> None:
        events = [
            {"Action": "pass", "Package": "p", "Test": "TestOK", "Elapsed": 0.001},
            {"Action": "pass", "Package": "p", "Elapsed": 0.5},
        ]
        json_path = (tmp_path / "results.jsonl").resolve()
        _write_jsonl(json_path, events)
        test_files = _make_test_files(tmp_path)
        cfg = _make_test_config(tmp_path)

        results = parse_go_test_output(json_path, test_files, cfg)
        assert len(results.test_results) == 1


class TestParseGoTestOutputBenchmark:
    def test_benchmark_parsing(self, tmp_path: Path) -> None:
        events = [
            {"Action": "run", "Package": "p", "Test": "BenchmarkAdd"},
            {"Action": "output", "Package": "p", "Test": "BenchmarkAdd", "Output": "BenchmarkAdd-8   \t 1000000\t      1234 ns/op\t      56 B/op\t       2 allocs/op\n"},
            {"Action": "pass", "Package": "p", "Test": "BenchmarkAdd", "Elapsed": 1.5},
        ]
        json_path = (tmp_path / "bench.jsonl").resolve()
        _write_jsonl(json_path, events)
        test_files = _make_test_files(tmp_path)
        cfg = _make_test_config(tmp_path)

        results = parse_go_test_output(json_path, test_files, cfg)
        assert len(results.test_results) == 1
        result = results.test_results[0]
        assert result.did_pass is True
        assert result.runtime == 1234

    def test_benchmark_overrides_elapsed(self, tmp_path: Path) -> None:
        events = [
            {"Action": "output", "Package": "p", "Test": "BenchmarkSort", "Output": "BenchmarkSort   5000   300000 ns/op\n"},
            {"Action": "pass", "Package": "p", "Test": "BenchmarkSort", "Elapsed": 2.0},
        ]
        json_path = (tmp_path / "bench.jsonl").resolve()
        _write_jsonl(json_path, events)
        test_files = _make_test_files(tmp_path)
        cfg = _make_test_config(tmp_path)

        results = parse_go_test_output(json_path, test_files, cfg)
        assert results.test_results[0].runtime == 300000

    def test_mixed_tests_and_benchmarks(self, tmp_path: Path) -> None:
        events = [
            {"Action": "pass", "Package": "p", "Test": "TestAdd", "Elapsed": 0.001},
            {"Action": "output", "Package": "p", "Test": "BenchmarkAdd", "Output": "BenchmarkAdd-8   1000000   500 ns/op\n"},
            {"Action": "pass", "Package": "p", "Test": "BenchmarkAdd", "Elapsed": 1.0},
        ]
        json_path = (tmp_path / "mixed.jsonl").resolve()
        _write_jsonl(json_path, events)
        test_files = _make_test_files(tmp_path)
        cfg = _make_test_config(tmp_path)

        results = parse_go_test_output(json_path, test_files, cfg)
        by_name = {r.id.test_function_name: r for r in results.test_results}

        assert by_name["TestAdd"].runtime == 1_000_000
        assert by_name["BenchmarkAdd"].runtime == 500


class TestParseGoTestOutputInvocationId:
    def test_invocation_id_fields(self, tmp_path: Path) -> None:
        events = [
            {"Action": "pass", "Package": "example/calc", "Test": "TestAdd", "Elapsed": 0.001},
        ]
        json_path = (tmp_path / "results.jsonl").resolve()
        _write_jsonl(json_path, events)
        test_files = _make_test_files(tmp_path)
        cfg = _make_test_config(tmp_path)

        results = parse_go_test_output(json_path, test_files, cfg)
        inv = results.test_results[0]
        assert inv.id.test_module_path == "example/calc"
        assert inv.id.test_class_name is None
        assert inv.id.test_function_name == "TestAdd"
        assert inv.loop_index == 1

    def test_unique_invocation_loop_id_stable(self, tmp_path: Path) -> None:
        events = [
            {"Action": "pass", "Package": "p", "Test": "TestA", "Elapsed": 0.001},
        ]
        json_path = (tmp_path / "results.jsonl").resolve()
        _write_jsonl(json_path, events)
        test_files = _make_test_files(tmp_path)
        cfg = _make_test_config(tmp_path)

        results1 = parse_go_test_output(json_path, test_files, cfg)
        results2 = parse_go_test_output(json_path, test_files, cfg)

        id1 = results1.test_results[0].unique_invocation_loop_id
        id2 = results2.test_results[0].unique_invocation_loop_id
        assert id1 == id2


class TestParseGoTestOutputComparison:
    def test_behavioral_comparison_same_results(self, tmp_path: Path) -> None:
        from codeflash.verification.equivalence import compare_test_results

        events = [
            {"Action": "pass", "Package": "p", "Test": "TestAdd", "Elapsed": 0.001},
            {"Action": "pass", "Package": "p", "Test": "TestSub", "Elapsed": 0.002},
        ]
        json_path = (tmp_path / "results.jsonl").resolve()
        _write_jsonl(json_path, events)
        test_files = _make_test_files(tmp_path)
        cfg = _make_test_config(tmp_path)

        original = parse_go_test_output(json_path, test_files, cfg)
        candidate = parse_go_test_output(json_path, test_files, cfg)

        are_equal, diffs = compare_test_results(original, candidate, pass_fail_only=True)
        assert are_equal is True
        assert diffs == []

    def test_behavioral_comparison_different_results(self, tmp_path: Path) -> None:
        from codeflash.verification.equivalence import compare_test_results

        original_events = [
            {"Action": "pass", "Package": "p", "Test": "TestAdd", "Elapsed": 0.001},
        ]
        candidate_events = [
            {"Action": "fail", "Package": "p", "Test": "TestAdd", "Elapsed": 0.001},
        ]
        orig_path = (tmp_path / "orig.jsonl").resolve()
        cand_path = (tmp_path / "cand.jsonl").resolve()
        _write_jsonl(orig_path, original_events)
        _write_jsonl(cand_path, candidate_events)

        test_files = _make_test_files(tmp_path)
        cfg = _make_test_config(tmp_path)

        original = parse_go_test_output(orig_path, test_files, cfg)
        candidate = parse_go_test_output(cand_path, test_files, cfg)

        are_equal, diffs = compare_test_results(original, candidate, pass_fail_only=True)
        assert are_equal is False
        assert len(diffs) == 1

    def test_empty_results_not_equal(self, tmp_path: Path) -> None:
        from codeflash.models.models import TestResults
        from codeflash.verification.equivalence import compare_test_results

        are_equal, _diffs = compare_test_results(TestResults(), TestResults(), pass_fail_only=True)
        assert are_equal is False
