from __future__ import annotations

from pathlib import Path

from codeflash.languages.golang.test_runner import (
    _build_bench_regex,
    _build_run_regex,
    _collect_other_test_files,
    _deduplicate_test_func_names,
    _deduplicated_test_files,
    _extract_func_names,
    _hide_other_test_files,
    _test_files_to_packages,
    _BENCH_FUNC_RE,
    _TEST_FUNC_RE,
    parse_go_test_json,
    parse_test_results,
)


GO_TEST_JSON_ALL_PASS = """\
{"Time":"2024-01-01T00:00:00Z","Action":"run","Package":"example.com/calc","Test":"TestAdd"}
{"Time":"2024-01-01T00:00:00Z","Action":"output","Package":"example.com/calc","Test":"TestAdd","Output":"=== RUN   TestAdd\\n"}
{"Time":"2024-01-01T00:00:00Z","Action":"output","Package":"example.com/calc","Test":"TestAdd","Output":"--- PASS: TestAdd (0.00s)\\n"}
{"Time":"2024-01-01T00:00:00Z","Action":"pass","Package":"example.com/calc","Test":"TestAdd","Elapsed":0.001}
{"Time":"2024-01-01T00:00:00Z","Action":"run","Package":"example.com/calc","Test":"TestSub"}
{"Time":"2024-01-01T00:00:00Z","Action":"output","Package":"example.com/calc","Test":"TestSub","Output":"--- PASS: TestSub (0.00s)\\n"}
{"Time":"2024-01-01T00:00:00Z","Action":"pass","Package":"example.com/calc","Test":"TestSub","Elapsed":0.002}
"""

GO_TEST_JSON_WITH_FAILURE = """\
{"Time":"2024-01-01T00:00:00Z","Action":"run","Package":"example.com/calc","Test":"TestAdd"}
{"Time":"2024-01-01T00:00:00Z","Action":"pass","Package":"example.com/calc","Test":"TestAdd","Elapsed":0.001}
{"Time":"2024-01-01T00:00:00Z","Action":"run","Package":"example.com/calc","Test":"TestBroken"}
{"Time":"2024-01-01T00:00:00Z","Action":"output","Package":"example.com/calc","Test":"TestBroken","Output":"    calc_test.go:15: expected 5, got 3\\n"}
{"Time":"2024-01-01T00:00:00Z","Action":"fail","Package":"example.com/calc","Test":"TestBroken","Elapsed":0.003}
"""


class TestParseGoTestJson:
    def test_all_pass(self) -> None:
        results = parse_go_test_json(GO_TEST_JSON_ALL_PASS)
        assert len(results) == 2
        by_name = {r.test_name: r for r in results}
        assert by_name["TestAdd"].passed is True
        assert by_name["TestAdd"].runtime_ns == 1_000_000
        assert by_name["TestSub"].passed is True
        assert by_name["TestSub"].runtime_ns == 2_000_000

    def test_with_failure(self) -> None:
        results = parse_go_test_json(GO_TEST_JSON_WITH_FAILURE)
        assert len(results) == 2
        by_name = {r.test_name: r for r in results}
        assert by_name["TestAdd"].passed is True
        assert by_name["TestBroken"].passed is False
        assert by_name["TestBroken"].error_message == "Test TestBroken failed"

    def test_empty_input(self) -> None:
        results = parse_go_test_json("")
        assert results == []

    def test_invalid_json_lines_skipped(self) -> None:
        json_output = 'not json\n{"Action":"pass","Package":"calc","Test":"TestOk","Elapsed":0.001}\n'
        results = parse_go_test_json(json_output)
        assert len(results) == 1
        assert results[0].test_name == "TestOk"
        assert results[0].passed is True

    def test_package_level_events_ignored(self) -> None:
        json_output = '{"Action":"pass","Package":"example.com/calc","Elapsed":0.5}\n'
        results = parse_go_test_json(json_output)
        assert results == []

    def test_runtime_ns_conversion(self) -> None:
        json_output = '{"Action":"pass","Package":"calc","Test":"TestFast","Elapsed":0.0005}\n'
        results = parse_go_test_json(json_output)
        assert len(results) == 1
        assert results[0].runtime_ns == 500_000

    def test_zero_elapsed(self) -> None:
        json_output = '{"Action":"pass","Package":"calc","Test":"TestZero","Elapsed":0}\n'
        results = parse_go_test_json(json_output)
        assert len(results) == 1
        assert results[0].runtime_ns is None


class TestParseTestResults:
    def test_reads_from_file(self, tmp_path: Path) -> None:
        json_file = (tmp_path / "results.jsonl").resolve()
        json_file.write_text('{"Action":"pass","Package":"calc","Test":"TestAdd","Elapsed":0.001}\n', encoding="utf-8")
        results = parse_test_results(json_file, "")
        assert len(results) == 1
        assert results[0].test_name == "TestAdd"
        assert results[0].passed is True

    def test_falls_back_to_stdout(self, tmp_path: Path) -> None:
        missing_file = (tmp_path / "missing.jsonl").resolve()
        stdout = '{"Action":"fail","Package":"calc","Test":"TestBad","Elapsed":0.002}\n'
        results = parse_test_results(missing_file, stdout)
        assert len(results) == 1
        assert results[0].test_name == "TestBad"
        assert results[0].passed is False


class TestCollectOtherTestFiles:
    def test_finds_other_test_files_in_same_dir(self, tmp_path: Path) -> None:
        keep = (tmp_path / "instrumented_test.go").resolve()
        keep.write_text("package x", encoding="utf-8")
        other1 = (tmp_path / "sorting_test.go").resolve()
        other1.write_text("package x", encoding="utf-8")
        other2 = (tmp_path / "perf_test.go").resolve()
        other2.write_text("package x", encoding="utf-8")

        result = _collect_other_test_files([keep])
        resolved = {f.resolve() for f in result}
        assert other1.resolve() in resolved
        assert other2.resolve() in resolved
        assert keep.resolve() not in resolved

    def test_keeps_only_specified_files(self, tmp_path: Path) -> None:
        f1 = (tmp_path / "a_test.go").resolve()
        f1.write_text("package x", encoding="utf-8")
        f2 = (tmp_path / "b_test.go").resolve()
        f2.write_text("package x", encoding="utf-8")

        result = _collect_other_test_files([f1, f2])
        assert result == []

    def test_ignores_non_test_files(self, tmp_path: Path) -> None:
        keep = (tmp_path / "target_test.go").resolve()
        keep.write_text("package x", encoding="utf-8")
        non_test = (tmp_path / "helper.go").resolve()
        non_test.write_text("package x", encoding="utf-8")

        result = _collect_other_test_files([keep])
        assert all(f.name.endswith("_test.go") for f in result)
        assert non_test not in result

    def test_empty_list(self) -> None:
        assert _collect_other_test_files([]) == []

    def test_multiple_dirs(self, tmp_path: Path) -> None:
        d1 = (tmp_path / "pkg1").resolve()
        d1.mkdir()
        d2 = (tmp_path / "pkg2").resolve()
        d2.mkdir()
        keep1 = (d1 / "target_test.go").resolve()
        keep1.write_text("package pkg1", encoding="utf-8")
        other1 = (d1 / "old_test.go").resolve()
        other1.write_text("package pkg1", encoding="utf-8")
        keep2 = (d2 / "target_test.go").resolve()
        keep2.write_text("package pkg2", encoding="utf-8")

        result = _collect_other_test_files([keep1, keep2])
        resolved = {f.resolve() for f in result}
        assert other1.resolve() in resolved
        assert keep1.resolve() not in resolved
        assert keep2.resolve() not in resolved


class TestHideOtherTestFiles:
    def test_hides_and_restores(self, tmp_path: Path) -> None:
        other = (tmp_path / "sorting_test.go").resolve()
        other.write_text("package x\n\nfunc TestSort(t *testing.T) {}", encoding="utf-8")

        with _hide_other_test_files([other]):
            assert not other.exists()
            assert other.with_suffix(".go.codeflash_hidden").exists()

        assert other.exists()
        assert not other.with_suffix(".go.codeflash_hidden").exists()
        assert other.read_text(encoding="utf-8") == "package x\n\nfunc TestSort(t *testing.T) {}"

    def test_restores_even_on_exception(self, tmp_path: Path) -> None:
        other = (tmp_path / "sorting_test.go").resolve()
        other.write_text("content", encoding="utf-8")

        try:
            with _hide_other_test_files([other]):
                raise RuntimeError("boom")
        except RuntimeError:
            pass

        assert other.exists()
        assert not other.with_suffix(".go.codeflash_hidden").exists()

    def test_empty_list_is_noop(self) -> None:
        with _hide_other_test_files([]):
            pass

    def test_multiple_files(self, tmp_path: Path) -> None:
        files = []
        for name in ("a_test.go", "b_test.go"):
            f = (tmp_path / name).resolve()
            f.write_text(f"package {name}", encoding="utf-8")
            files.append(f)

        with _hide_other_test_files(files):
            for f in files:
                assert not f.exists()

        for f in files:
            assert f.exists()


GO_TEST_SOURCE = """\
package sorting

import "testing"

func TestBubbleSort_Basic(t *testing.T) {}
func TestBubbleSort_EdgeCases(t *testing.T) {}
"""

GO_BENCH_SOURCE = """\
package sorting

import "testing"

func BenchmarkBubbleSort(b *testing.B) {}
func BenchmarkBubbleSort_Large(b *testing.B) {}
"""

GO_MIXED_SOURCE = """\
package sorting

import "testing"

func TestBubbleSort(t *testing.T) {}
func BenchmarkBubbleSort(b *testing.B) {}
"""


class TestExtractFuncNames:
    def test_extracts_test_funcs(self, tmp_path: Path) -> None:
        f = (tmp_path / "sorting_test.go").resolve()
        f.write_text(GO_TEST_SOURCE, encoding="utf-8")
        names = _extract_func_names([f], _TEST_FUNC_RE)
        assert names == ["TestBubbleSort_Basic", "TestBubbleSort_EdgeCases"]

    def test_extracts_bench_funcs(self, tmp_path: Path) -> None:
        f = (tmp_path / "sorting_test.go").resolve()
        f.write_text(GO_BENCH_SOURCE, encoding="utf-8")
        names = _extract_func_names([f], _BENCH_FUNC_RE)
        assert names == ["BenchmarkBubbleSort", "BenchmarkBubbleSort_Large"]

    def test_test_regex_does_not_match_benchmarks(self, tmp_path: Path) -> None:
        f = (tmp_path / "sorting_test.go").resolve()
        f.write_text(GO_BENCH_SOURCE, encoding="utf-8")
        names = _extract_func_names([f], _TEST_FUNC_RE)
        assert names == []

    def test_multiple_files(self, tmp_path: Path) -> None:
        f1 = (tmp_path / "a_test.go").resolve()
        f1.write_text("package x\nfunc TestA(t *testing.T) {}", encoding="utf-8")
        f2 = (tmp_path / "b_test.go").resolve()
        f2.write_text("package x\nfunc TestB(t *testing.T) {}", encoding="utf-8")
        names = _extract_func_names([f1, f2], _TEST_FUNC_RE)
        assert names == ["TestA", "TestB"]

    def test_missing_file_skipped(self, tmp_path: Path) -> None:
        missing = (tmp_path / "missing_test.go").resolve()
        names = _extract_func_names([missing], _TEST_FUNC_RE)
        assert names == []

    def test_empty_list(self) -> None:
        assert _extract_func_names([], _TEST_FUNC_RE) == []


class TestBuildRunRegex:
    def test_single_test_func(self, tmp_path: Path) -> None:
        f = (tmp_path / "a_test.go").resolve()
        f.write_text("package x\nfunc TestFoo(t *testing.T) {}", encoding="utf-8")
        regex = _build_run_regex([f])
        assert regex == "^(TestFoo)$"

    def test_multiple_test_funcs(self, tmp_path: Path) -> None:
        f = (tmp_path / "a_test.go").resolve()
        f.write_text(GO_TEST_SOURCE, encoding="utf-8")
        regex = _build_run_regex([f])
        assert regex == "^(TestBubbleSort_Basic|TestBubbleSort_EdgeCases)$"

    def test_no_test_funcs_returns_none(self, tmp_path: Path) -> None:
        f = (tmp_path / "a_test.go").resolve()
        f.write_text("package x\nfunc helper() {}", encoding="utf-8")
        assert _build_run_regex([f]) is None

    def test_empty_files_returns_none(self) -> None:
        assert _build_run_regex([]) is None


class TestBuildBenchRegex:
    def test_single_bench_func(self, tmp_path: Path) -> None:
        f = (tmp_path / "a_test.go").resolve()
        f.write_text('package x\nimport "testing"\nfunc BenchmarkFoo(b *testing.B) {}', encoding="utf-8")
        regex = _build_bench_regex([f])
        assert regex == "^(BenchmarkFoo)$"

    def test_no_bench_funcs_returns_none(self, tmp_path: Path) -> None:
        f = (tmp_path / "a_test.go").resolve()
        f.write_text(GO_TEST_SOURCE, encoding="utf-8")
        assert _build_bench_regex([f]) is None


class TestTestFilesToPackages:
    def test_subdirectory(self, tmp_path: Path) -> None:
        subdir = (tmp_path / "sorting").resolve()
        subdir.mkdir()
        f = subdir / "sorting_test.go"
        f.write_text("package sorting", encoding="utf-8")
        packages = _test_files_to_packages([f.resolve()], tmp_path.resolve())
        assert packages == ["./sorting"]

    def test_root_directory(self, tmp_path: Path) -> None:
        f = (tmp_path / "main_test.go").resolve()
        f.write_text("package main", encoding="utf-8")
        packages = _test_files_to_packages([f], tmp_path.resolve())
        assert packages == ["."]

    def test_deduplicates_same_package(self, tmp_path: Path) -> None:
        f1 = (tmp_path / "a_test.go").resolve()
        f1.write_text("package x", encoding="utf-8")
        f2 = (tmp_path / "b_test.go").resolve()
        f2.write_text("package x", encoding="utf-8")
        packages = _test_files_to_packages([f1, f2], tmp_path.resolve())
        assert packages == ["."]

    def test_multiple_packages(self, tmp_path: Path) -> None:
        for name in ("pkg1", "pkg2"):
            d = (tmp_path / name).resolve()
            d.mkdir()
            (d / "x_test.go").write_text(f"package {name}", encoding="utf-8")
        f1 = (tmp_path / "pkg1" / "x_test.go").resolve()
        f2 = (tmp_path / "pkg2" / "x_test.go").resolve()
        packages = _test_files_to_packages([f1, f2], tmp_path.resolve())
        assert packages == ["./pkg1", "./pkg2"]

    def test_empty_list(self, tmp_path: Path) -> None:
        assert _test_files_to_packages([], tmp_path.resolve()) == []

    def test_file_outside_cwd_skipped(self, tmp_path: Path) -> None:
        other = (tmp_path / "other").resolve()
        other.mkdir()
        f = (other / "x_test.go").resolve()
        f.write_text("package x", encoding="utf-8")
        cwd = (tmp_path / "project").resolve()
        cwd.mkdir()
        assert _test_files_to_packages([f], cwd) == []


GO_FILE_A = """\
package x

import "testing"

func TestFoo(t *testing.T) {}
func TestBar(t *testing.T) {}
"""

GO_FILE_B_DUPLICATES = """\
package x

import "testing"

func TestFoo(t *testing.T) {}
func TestBaz(t *testing.T) {}
"""

GO_FILE_C_MORE_DUPLICATES = """\
package x

import "testing"

func TestFoo(t *testing.T) {}
func TestBar(t *testing.T) {}
func TestNew(t *testing.T) {}
"""


class TestDeduplicateTestFuncNames:
    def test_no_duplicates_no_changes(self, tmp_path: Path) -> None:
        f1 = (tmp_path / "a_test.go").resolve()
        f1.write_text(GO_FILE_A, encoding="utf-8")
        originals = _deduplicate_test_func_names([f1])
        assert originals == {}
        assert f1.read_text(encoding="utf-8") == GO_FILE_A

    def test_renames_duplicates_in_second_file(self, tmp_path: Path) -> None:
        f1 = (tmp_path / "a_test.go").resolve()
        f1.write_text(GO_FILE_A, encoding="utf-8")
        f2 = (tmp_path / "b_test.go").resolve()
        f2.write_text(GO_FILE_B_DUPLICATES, encoding="utf-8")

        originals = _deduplicate_test_func_names([f1, f2])

        assert f1.read_text(encoding="utf-8") == GO_FILE_A
        assert f2 in originals
        assert originals[f2] == GO_FILE_B_DUPLICATES

        rewritten = f2.read_text(encoding="utf-8")
        assert "func TestFoo_1(" in rewritten
        assert "func TestBaz(" in rewritten
        assert "func TestFoo(" not in rewritten

    def test_renames_across_three_files(self, tmp_path: Path) -> None:
        f1 = (tmp_path / "a_test.go").resolve()
        f1.write_text(GO_FILE_A, encoding="utf-8")
        f2 = (tmp_path / "b_test.go").resolve()
        f2.write_text(GO_FILE_B_DUPLICATES, encoding="utf-8")
        f3 = (tmp_path / "c_test.go").resolve()
        f3.write_text(GO_FILE_C_MORE_DUPLICATES, encoding="utf-8")

        _deduplicate_test_func_names([f1, f2, f3])

        rewritten_b = f2.read_text(encoding="utf-8")
        rewritten_c = f3.read_text(encoding="utf-8")

        assert "func TestFoo_1(" in rewritten_b
        assert "func TestFoo_2(" in rewritten_c
        assert "func TestBar_1(" in rewritten_c

    def test_empty_list(self) -> None:
        assert _deduplicate_test_func_names([]) == {}

    def test_single_file_no_changes(self, tmp_path: Path) -> None:
        f = (tmp_path / "a_test.go").resolve()
        f.write_text(GO_FILE_B_DUPLICATES, encoding="utf-8")
        originals = _deduplicate_test_func_names([f])
        assert originals == {}

    def test_benchmarks_also_deduplicated(self, tmp_path: Path) -> None:
        f1 = (tmp_path / "a_test.go").resolve()
        f1.write_text('package x\n\nimport "testing"\n\nfunc BenchmarkFoo(b *testing.B) {}\n', encoding="utf-8")
        f2 = (tmp_path / "b_test.go").resolve()
        f2.write_text('package x\n\nimport "testing"\n\nfunc BenchmarkFoo(b *testing.B) {}\n', encoding="utf-8")

        _deduplicate_test_func_names([f1, f2])

        assert "func BenchmarkFoo(" in f1.read_text(encoding="utf-8")
        rewritten = f2.read_text(encoding="utf-8")
        assert "func BenchmarkFoo_1(" in rewritten


class TestDeduplicatedTestFiles:
    def test_restores_after_context(self, tmp_path: Path) -> None:
        f1 = (tmp_path / "a_test.go").resolve()
        f1.write_text(GO_FILE_A, encoding="utf-8")
        f2 = (tmp_path / "b_test.go").resolve()
        f2.write_text(GO_FILE_B_DUPLICATES, encoding="utf-8")

        with _deduplicated_test_files([f1, f2]):
            assert "func TestFoo_1(" in f2.read_text(encoding="utf-8")

        assert f2.read_text(encoding="utf-8") == GO_FILE_B_DUPLICATES

    def test_restores_on_exception(self, tmp_path: Path) -> None:
        f1 = (tmp_path / "a_test.go").resolve()
        f1.write_text(GO_FILE_A, encoding="utf-8")
        f2 = (tmp_path / "b_test.go").resolve()
        f2.write_text(GO_FILE_B_DUPLICATES, encoding="utf-8")

        try:
            with _deduplicated_test_files([f1, f2]):
                raise RuntimeError("boom")
        except RuntimeError:
            pass

        assert f2.read_text(encoding="utf-8") == GO_FILE_B_DUPLICATES

    def test_no_duplicates_is_noop(self, tmp_path: Path) -> None:
        f = (tmp_path / "a_test.go").resolve()
        f.write_text(GO_FILE_A, encoding="utf-8")

        with _deduplicated_test_files([f]):
            assert f.read_text(encoding="utf-8") == GO_FILE_A
