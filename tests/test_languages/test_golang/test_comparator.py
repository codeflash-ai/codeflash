from __future__ import annotations

from pathlib import Path

from codeflash.languages.golang.comparator import compare_test_results


class TestCompareTestResults:
    def test_equivalent_results(self, tmp_path: Path) -> None:
        orig = (tmp_path / "original.jsonl").resolve()
        cand = (tmp_path / "candidate.jsonl").resolve()
        orig.write_text(
            '{"Action":"pass","Test":"TestAdd","Package":"calc"}\n'
            '{"Action":"pass","Test":"TestSub","Package":"calc"}\n',
            encoding="utf-8",
        )
        cand.write_text(
            '{"Action":"pass","Test":"TestAdd","Package":"calc"}\n'
            '{"Action":"pass","Test":"TestSub","Package":"calc"}\n',
            encoding="utf-8",
        )
        eq, diffs = compare_test_results(orig, cand)
        assert eq is True
        assert diffs == []

    def test_candidate_fails_one(self, tmp_path: Path) -> None:
        orig = (tmp_path / "original.jsonl").resolve()
        cand = (tmp_path / "candidate.jsonl").resolve()
        orig.write_text(
            '{"Action":"pass","Test":"TestAdd","Package":"calc"}\n'
            '{"Action":"pass","Test":"TestSub","Package":"calc"}\n',
            encoding="utf-8",
        )
        cand.write_text(
            '{"Action":"pass","Test":"TestAdd","Package":"calc"}\n'
            '{"Action":"fail","Test":"TestSub","Package":"calc"}\n',
            encoding="utf-8",
        )
        eq, diffs = compare_test_results(orig, cand)
        assert eq is False
        assert len(diffs) == 1
        assert diffs[0].test_name == "TestSub"
        assert diffs[0].original_passed is True
        assert diffs[0].candidate_passed is False

    def test_missing_test_in_candidate(self, tmp_path: Path) -> None:
        orig = (tmp_path / "original.jsonl").resolve()
        cand = (tmp_path / "candidate.jsonl").resolve()
        orig.write_text(
            '{"Action":"pass","Test":"TestAdd","Package":"calc"}\n'
            '{"Action":"pass","Test":"TestSub","Package":"calc"}\n',
            encoding="utf-8",
        )
        cand.write_text(
            '{"Action":"pass","Test":"TestAdd","Package":"calc"}\n',
            encoding="utf-8",
        )
        eq, diffs = compare_test_results(orig, cand)
        assert eq is False
        assert len(diffs) == 1
        assert diffs[0].test_name == "TestSub"

    def test_extra_test_in_candidate(self, tmp_path: Path) -> None:
        orig = (tmp_path / "original.jsonl").resolve()
        cand = (tmp_path / "candidate.jsonl").resolve()
        orig.write_text(
            '{"Action":"pass","Test":"TestAdd","Package":"calc"}\n',
            encoding="utf-8",
        )
        cand.write_text(
            '{"Action":"pass","Test":"TestAdd","Package":"calc"}\n'
            '{"Action":"pass","Test":"TestNew","Package":"calc"}\n',
            encoding="utf-8",
        )
        eq, diffs = compare_test_results(orig, cand)
        assert eq is False
        assert len(diffs) == 1
        assert diffs[0].test_name == "TestNew"

    def test_both_empty(self, tmp_path: Path) -> None:
        orig = (tmp_path / "original.jsonl").resolve()
        cand = (tmp_path / "candidate.jsonl").resolve()
        orig.write_text("", encoding="utf-8")
        cand.write_text("", encoding="utf-8")
        eq, diffs = compare_test_results(orig, cand)
        assert eq is True
        assert diffs == []

    def test_missing_files(self, tmp_path: Path) -> None:
        orig = (tmp_path / "missing1.jsonl").resolve()
        cand = (tmp_path / "missing2.jsonl").resolve()
        eq, diffs = compare_test_results(orig, cand)
        assert eq is True
        assert diffs == []

    def test_both_fail_same_test(self, tmp_path: Path) -> None:
        orig = (tmp_path / "original.jsonl").resolve()
        cand = (tmp_path / "candidate.jsonl").resolve()
        orig.write_text(
            '{"Action":"fail","Test":"TestBroken","Package":"calc"}\n',
            encoding="utf-8",
        )
        cand.write_text(
            '{"Action":"fail","Test":"TestBroken","Package":"calc"}\n',
            encoding="utf-8",
        )
        eq, diffs = compare_test_results(orig, cand)
        assert eq is True
        assert diffs == []
