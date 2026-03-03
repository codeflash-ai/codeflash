"""Tests for compilation failure recovery helpers in the Java test runner.

These helpers parse Maven error output to detect failing generated test files,
remove them from disk, and filter them out of test_paths objects, enabling
a retry without the problematic files.
"""

from __future__ import annotations

from pathlib import Path

from codeflash.languages.java.test_runner import (
    _extract_failing_java_files,
    _filter_test_paths_excluding_files,
    _remove_failing_generated_tests,
)

# ---------------------------------------------------------------------------
# _extract_failing_java_files
# ---------------------------------------------------------------------------


class TestExtractFailingJavaFiles:
    """Tests for _extract_failing_java_files."""

    def test_empty_output(self) -> None:
        """No files in empty output."""
        assert _extract_failing_java_files("") == set()

    def test_single_error(self) -> None:
        """Single Maven compilation error parsed correctly."""
        output = (
            "[ERROR] /home/user/project/test/src/FooTest__perfinstrumented.java:[42,13] "
            "cannot find symbol\n"
        )
        result = _extract_failing_java_files(output)
        assert result == {Path("/home/user/project/test/src/FooTest__perfinstrumented.java")}

    def test_multiple_errors_same_file(self) -> None:
        """Multiple errors in the same file → only one entry returned."""
        output = (
            "[ERROR] /home/user/Foo__perfinstrumented.java:[10,5] error: cannot find symbol\n"
            "[ERROR] /home/user/Foo__perfinstrumented.java:[20,5] error: package x does not exist\n"
        )
        result = _extract_failing_java_files(output)
        assert result == {Path("/home/user/Foo__perfinstrumented.java")}

    def test_multiple_different_files(self) -> None:
        """Multiple distinct failing files all collected."""
        output = (
            "[ERROR] /a/Foo__perfinstrumented.java:[1,1] bad stuff\n"
            "[ERROR] /b/Bar__perfonlyinstrumented.java:[2,2] bad stuff\n"
            "[ERROR] /c/Baz__perfinstrumented.java:[3,3] bad stuff\n"
        )
        result = _extract_failing_java_files(output)
        assert result == {
            Path("/a/Foo__perfinstrumented.java"),
            Path("/b/Bar__perfonlyinstrumented.java"),
            Path("/c/Baz__perfinstrumented.java"),
        }

    def test_non_java_errors_ignored(self) -> None:
        """Lines that are not Java file compilation errors are ignored."""
        output = (
            "[ERROR] Some other Maven error\n"
            "[ERROR] BUILD FAILURE\n"
            "[INFO] This is not an error\n"
        )
        assert _extract_failing_java_files(output) == set()

    def test_ignores_non_generated_files(self) -> None:
        """Non-generated .java files are returned too (caller decides what to remove)."""
        output = "[ERROR] /src/main/java/Regular.java:[5,3] error\n"
        result = _extract_failing_java_files(output)
        # _extract returns ALL failing files; _remove only deletes generated ones
        assert result == {Path("/src/main/java/Regular.java")}

    def test_column_zero(self) -> None:
        """Column 0 is a valid position."""
        output = "[ERROR] /home/user/Test__perfinstrumented.java:[0,0] error\n"
        result = _extract_failing_java_files(output)
        assert result == {Path("/home/user/Test__perfinstrumented.java")}

    def test_path_with_spaces_not_matched(self) -> None:
        """Paths with spaces in them would break the regex (by design — Maven doesn't use them)."""
        output = "[ERROR] /home/user/path with spaces/Foo.java:[1,1] error\n"
        # The regex stops at whitespace so the path would be wrong — it's fine,
        # we just check it doesn't crash.
        result = _extract_failing_java_files(output)
        assert isinstance(result, set)


# ---------------------------------------------------------------------------
# _remove_failing_generated_tests
# ---------------------------------------------------------------------------


class TestRemoveFailingGeneratedTests:
    """Tests for _remove_failing_generated_tests."""

    def test_empty_set(self) -> None:
        """Empty input → nothing removed."""
        assert _remove_failing_generated_tests(set()) == []

    def test_removes_perfinstrumented_file(self, tmp_path: Path) -> None:
        """Files with __perfinstrumented in the name are deleted."""
        f = tmp_path / "Foo__perfinstrumented.java"
        f.write_text("// generated\n")
        removed = _remove_failing_generated_tests({f})
        assert removed == [f]
        assert not f.exists()

    def test_removes_perfonlyinstrumented_file(self, tmp_path: Path) -> None:
        """Files with __perfonlyinstrumented in the name are deleted."""
        f = tmp_path / "Foo__perfonlyinstrumented.java"
        f.write_text("// generated\n")
        removed = _remove_failing_generated_tests({f})
        assert removed == [f]
        assert not f.exists()

    def test_does_not_remove_regular_files(self, tmp_path: Path) -> None:
        """Regular (non-generated) files are never removed."""
        f = tmp_path / "RegularTest.java"
        f.write_text("// real test\n")
        removed = _remove_failing_generated_tests({f})
        assert removed == []
        assert f.exists()

    def test_does_not_remove_nonexistent_files(self, tmp_path: Path) -> None:
        """Non-existent generated files don't raise — just skipped."""
        f = tmp_path / "Ghost__perfinstrumented.java"
        assert not f.exists()
        removed = _remove_failing_generated_tests({f})
        assert removed == []

    def test_mixed_files(self, tmp_path: Path) -> None:
        """Only generated files among a mixed set are removed."""
        gen = tmp_path / "Gen__perfinstrumented.java"
        gen.write_text("// gen\n")
        real = tmp_path / "RealTest.java"
        real.write_text("// real\n")

        removed = _remove_failing_generated_tests({gen, real})
        # Only the generated file is removed (set input, so order may vary)
        assert set(removed) == {gen}
        assert not gen.exists()
        assert real.exists()


# ---------------------------------------------------------------------------
# _filter_test_paths_excluding_files
# ---------------------------------------------------------------------------


class TestFilterTestPathsExcludingFiles:
    """Tests for _filter_test_paths_excluding_files."""

    def test_empty_removed_files_returns_same(self) -> None:
        """If nothing was removed, test_paths is returned unchanged."""
        paths = [Path("/a"), Path("/b")]
        result = _filter_test_paths_excluding_files(paths, [])
        assert result is paths

    def test_filters_list_of_paths(self, tmp_path: Path) -> None:
        """A plain list of Path objects is filtered correctly."""
        a = tmp_path / "a.java"
        b = tmp_path / "b.java"
        a.write_text("")
        b.write_text("")
        result = _filter_test_paths_excluding_files([a, b], [a])
        assert result == [b]

    def test_filters_tuple_of_paths(self, tmp_path: Path) -> None:
        """A tuple of Path objects is filtered and returned as a tuple."""
        a = tmp_path / "a.java"
        b = tmp_path / "b.java"
        a.write_text("")
        b.write_text("")
        result = _filter_test_paths_excluding_files((a, b), [a])
        assert isinstance(result, tuple)
        assert result == (b,)

    def test_filters_testfiles_by_behavior_path(self, tmp_path: Path) -> None:
        """TestFiles entries whose behavior path matches are removed."""
        from codeflash.models.models import TestFile, TestFiles, TestType

        behavior = tmp_path / "Foo__perfinstrumented.java"
        bench = tmp_path / "Foo__perfonlyinstrumented.java"
        behavior.write_text("")
        bench.write_text("")

        tf = TestFile(
            instrumented_behavior_file_path=behavior,
            benchmarking_file_path=bench,
            original_file_path=None,
            original_source="",
            test_type=TestType.GENERATED_REGRESSION,
            tests_in_file=None,
        )
        test_paths = TestFiles(test_files=[tf])
        result = _filter_test_paths_excluding_files(test_paths, [behavior])
        assert len(result.test_files) == 0

    def test_filters_testfiles_by_bench_path(self, tmp_path: Path) -> None:
        """TestFiles entries whose bench path matches are removed."""
        from codeflash.models.models import TestFile, TestFiles, TestType

        behavior = tmp_path / "Foo__perfinstrumented.java"
        bench = tmp_path / "Foo__perfonlyinstrumented.java"
        behavior.write_text("")
        bench.write_text("")

        tf = TestFile(
            instrumented_behavior_file_path=behavior,
            benchmarking_file_path=bench,
            original_file_path=None,
            original_source="",
            test_type=TestType.GENERATED_REGRESSION,
            tests_in_file=None,
        )
        test_paths = TestFiles(test_files=[tf])
        result = _filter_test_paths_excluding_files(test_paths, [bench])
        assert len(result.test_files) == 0

    def test_keeps_unaffected_testfiles(self, tmp_path: Path) -> None:
        """TestFiles entries not matching removed files are kept."""
        from codeflash.models.models import TestFile, TestFiles, TestType

        good_behavior = tmp_path / "Good__perfinstrumented.java"
        bad_behavior = tmp_path / "Bad__perfinstrumented.java"
        good_behavior.write_text("")
        bad_behavior.write_text("")

        good_tf = TestFile(
            instrumented_behavior_file_path=good_behavior,
            benchmarking_file_path=None,
            original_file_path=None,
            original_source="",
            test_type=TestType.GENERATED_REGRESSION,
            tests_in_file=None,
        )
        bad_tf = TestFile(
            instrumented_behavior_file_path=bad_behavior,
            benchmarking_file_path=None,
            original_file_path=None,
            original_source="",
            test_type=TestType.GENERATED_REGRESSION,
            tests_in_file=None,
        )
        test_paths = TestFiles(test_files=[good_tf, bad_tf])
        result = _filter_test_paths_excluding_files(test_paths, [bad_behavior])
        assert len(result.test_files) == 1
        assert result.test_files[0].instrumented_behavior_file_path == good_behavior

    def test_unknown_type_returned_unchanged(self) -> None:
        """Unknown test_paths types are returned as-is."""
        obj = {"some": "dict"}
        result = _filter_test_paths_excluding_files(obj, [Path("/foo")])
        assert result is obj

    def test_path_resolution_used_for_comparison(self, tmp_path: Path) -> None:
        """Comparison uses resolved paths so relative vs absolute matches correctly."""
        f = tmp_path / "Foo__perfinstrumented.java"
        f.write_text("")
        # Pass a relative path equivalent (resolved in _filter by .resolve())
        result = _filter_test_paths_excluding_files([f], [f.resolve()])
        assert result == []
