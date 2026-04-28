from __future__ import annotations

from codeflash.languages.golang.instrumentation import _test_matches_target, convert_tests_to_benchmarks

SIMPLE_TEST = """\
package sample

import "testing"

func TestAdd(t *testing.T) {
\tgot := Add(1, 2)
\tif got != 3 {
\t\tt.Errorf("Add(1, 2) = %d, want 3", got)
\t}
}
"""

TEST_WITH_SUBTESTS = """\
package sample

import "testing"

func TestBubbleSort_BasicCases(t *testing.T) {
\ttests := []struct {
\t\tname  string
\t\tinput []int
\t\twant  []int
\t}{
\t\t{"sorted", []int{1, 2, 3}, []int{1, 2, 3}},
\t}
\tfor _, tt := range tests {
\t\tt.Run(tt.name, func(t *testing.T) {
\t\t\tgot := BubbleSort(tt.input)
\t\t\tif len(got) != len(tt.want) {
\t\t\t\tt.Errorf("wrong length")
\t\t\t}
\t\t})
\t}
}
"""

MULTIPLE_TESTS = """\
package sample

import "testing"

func TestFoo(t *testing.T) {
\tFoo()
}

func TestBar(t *testing.T) {
\tBar()
}
"""

BENCHMARK_ONLY = """\
package sample

import "testing"

func BenchmarkFoo(b *testing.B) {
\tfor i := 0; i < b.N; i++ {
\t\tFoo()
\t}
}
"""

TEST_WITH_HELPER = """\
package sample

import "testing"

func equalSlices(t *testing.T, got, want []int) {
\tif len(got) != len(want) {
\t\tt.Fatalf("length mismatch")
\t}
}

func TestBFS(t *testing.T) {
\tgot := BFS(graph, 0)
\tequalSlices(t, got, []int{0, 1, 2})
}
"""

TEST_WITH_PARALLEL = """\
package sample

import "testing"

func TestFoo(t *testing.T) {
\tt.Parallel()
\tFoo()
}

func TestBar(t *testing.T) {
\tt.Helper()
\tt.Parallel()
\tBar()
}
"""


class TestMatchesTarget:
    def test_exact_match(self) -> None:
        assert _test_matches_target("TestBFS", "BFS") is True

    def test_prefix_segment_match(self) -> None:
        assert _test_matches_target("TestBFS_BasicCases", "BFS") is True

    def test_suffix_segment_match(self) -> None:
        assert _test_matches_target("TestGraph_BFS", "BFS") is True

    def test_no_match_substring(self) -> None:
        assert _test_matches_target("TestBFSHelper", "BFS") is False

    def test_no_match_different_function(self) -> None:
        assert _test_matches_target("TestDFS", "BFS") is False

    def test_multi_underscore(self) -> None:
        assert _test_matches_target("TestBFS_Large_Graph", "BFS") is True


class TestConvertTestsToBenchmarks:
    def test_simple_test(self) -> None:
        result = convert_tests_to_benchmarks(SIMPLE_TEST, "Add")
        assert "func BenchmarkAdd(" in result
        assert "*testing.B)" in result
        assert "for i := 0; i < " in result
        assert ".N; i++ {" in result
        assert "func TestAdd(" not in result

    def test_subtests_converted(self) -> None:
        result = convert_tests_to_benchmarks(TEST_WITH_SUBTESTS, "BubbleSort")
        assert "func BenchmarkBubbleSort_BasicCases(" in result
        assert "*testing.T" not in result

    def test_multiple_functions_filtered(self) -> None:
        result = convert_tests_to_benchmarks(MULTIPLE_TESTS, "Foo")
        assert "func BenchmarkFoo(" in result
        assert "func BenchmarkBar(" not in result
        assert "func TestFoo(" not in result
        assert "func TestBar(" not in result

    def test_multiple_functions_no_filter(self) -> None:
        result = convert_tests_to_benchmarks(MULTIPLE_TESTS, "")
        assert "func BenchmarkFoo(" in result
        assert "func BenchmarkBar(" in result
        assert "func TestFoo(" not in result
        assert "func TestBar(" not in result

    def test_empty_source(self) -> None:
        assert convert_tests_to_benchmarks("", "Foo") == ""

    def test_no_test_functions(self) -> None:
        result = convert_tests_to_benchmarks(BENCHMARK_ONLY, "Foo")
        assert result == BENCHMARK_ONLY

    def test_package_preserved(self) -> None:
        result = convert_tests_to_benchmarks(SIMPLE_TEST, "Add")
        assert result.startswith("package sample")

    def test_import_preserved(self) -> None:
        result = convert_tests_to_benchmarks(SIMPLE_TEST, "Add")
        assert 'import "testing"' in result

    def test_helper_functions_converted(self) -> None:
        result = convert_tests_to_benchmarks(TEST_WITH_HELPER, "BFS")
        assert "func BenchmarkBFS(" in result
        assert "*testing.T" not in result
        assert "equalSlices" in result
        assert "*testing.B" in result

    def test_parallel_removed(self) -> None:
        result = convert_tests_to_benchmarks(TEST_WITH_PARALLEL, "Foo")
        assert ".Parallel()" not in result
        assert ".Helper()" not in result
        assert "func BenchmarkFoo(" in result
        assert "func BenchmarkBar(" not in result
