from __future__ import annotations

from pathlib import Path

from codeflash.languages.golang.test_runner import (
    _collect_original_file_paths,
    _hide_original_test_files,
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
        json_file.write_text(
            '{"Action":"pass","Package":"calc","Test":"TestAdd","Elapsed":0.001}\n',
            encoding="utf-8",
        )
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


class _FakeTestFile:
    def __init__(self, instrumented: Path | None = None, original: Path | None = None) -> None:
        self.instrumented_behavior_file_path = instrumented
        self.original_file_path = original


class _FakeTestFiles:
    def __init__(self, test_files: list[_FakeTestFile]) -> None:
        self.test_files = test_files


class TestCollectOriginalFilePaths:
    def test_returns_originals_when_instrumented_differs(self, tmp_path: Path) -> None:
        original = (tmp_path / "sorting_test.go").resolve()
        original.write_text("package x", encoding="utf-8")
        instrumented = (tmp_path / "sorting__perfinstrumented_test.go").resolve()
        tf = _FakeTestFile(instrumented=instrumented, original=original)
        result = _collect_original_file_paths(_FakeTestFiles([tf]))
        assert result == [original]

    def test_skips_when_same_path(self, tmp_path: Path) -> None:
        original = (tmp_path / "sorting_test.go").resolve()
        original.write_text("package x", encoding="utf-8")
        tf = _FakeTestFile(instrumented=original, original=original)
        result = _collect_original_file_paths(_FakeTestFiles([tf]))
        assert result == []

    def test_skips_missing_original(self, tmp_path: Path) -> None:
        original = (tmp_path / "missing_test.go").resolve()
        instrumented = (tmp_path / "missing__perfinstrumented_test.go").resolve()
        tf = _FakeTestFile(instrumented=instrumented, original=original)
        result = _collect_original_file_paths(_FakeTestFiles([tf]))
        assert result == []

    def test_none_test_paths(self) -> None:
        assert _collect_original_file_paths(None) == []


class TestHideOriginalTestFiles:
    def test_hides_and_restores(self, tmp_path: Path) -> None:
        original = (tmp_path / "sorting_test.go").resolve()
        original.write_text("package x\n\nfunc TestSort(t *testing.T) {}", encoding="utf-8")

        with _hide_original_test_files([original]):
            assert not original.exists()
            assert original.with_suffix(".go.codeflash_hidden").exists()

        assert original.exists()
        assert not original.with_suffix(".go.codeflash_hidden").exists()
        assert original.read_text(encoding="utf-8") == "package x\n\nfunc TestSort(t *testing.T) {}"

    def test_restores_even_on_exception(self, tmp_path: Path) -> None:
        original = (tmp_path / "sorting_test.go").resolve()
        original.write_text("content", encoding="utf-8")

        try:
            with _hide_original_test_files([original]):
                raise RuntimeError("boom")
        except RuntimeError:
            pass

        assert original.exists()
        assert not original.with_suffix(".go.codeflash_hidden").exists()

    def test_empty_list_is_noop(self) -> None:
        with _hide_original_test_files([]):
            pass

    def test_multiple_files(self, tmp_path: Path) -> None:
        files = []
        for name in ("a_test.go", "b_test.go"):
            f = (tmp_path / name).resolve()
            f.write_text(f"package {name}", encoding="utf-8")
            files.append(f)

        with _hide_original_test_files(files):
            for f in files:
                assert not f.exists()

        for f in files:
            assert f.exists()
