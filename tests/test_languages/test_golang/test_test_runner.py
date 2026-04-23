from __future__ import annotations

from pathlib import Path

from codeflash.languages.golang.test_runner import parse_go_test_json, parse_test_results


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
