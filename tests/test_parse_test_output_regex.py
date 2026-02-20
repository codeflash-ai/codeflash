"""Tests for the regex patterns and string matching in parse_test_output.py."""

from codeflash.verification.parse_test_output import (
    matches_re_end,
    matches_re_start,
    parse_test_failures_from_stdout,
)


# --- matches_re_start tests ---


class TestMatchesReStart:
    def test_simple_no_class(self) -> None:
        s = "!$######tests.test_foo:test_bar:target_func:1:abc######$!\n"
        m = matches_re_start.search(s)
        assert m is not None
        assert m.groups() == ("tests.test_foo", "", "test_bar", "target_func", "1", "abc")

    def test_with_class(self) -> None:
        s = "!$######tests.test_foo:MyClass.test_bar:target_func:1:abc######$!\n"
        m = matches_re_start.search(s)
        assert m is not None
        assert m.groups() == ("tests.test_foo", "MyClass.", "test_bar", "target_func", "1", "abc")

    def test_nested_class(self) -> None:
        s = "!$######a.b.c:A.B.test_x:func:3:id123######$!\n"
        m = matches_re_start.search(s)
        assert m is not None
        assert m.groups() == ("a.b.c", "A.B.", "test_x", "func", "3", "id123")

    def test_empty_class_and_function(self) -> None:
        s = "!$######mod::func:0:iter######$!\n"
        m = matches_re_start.search(s)
        assert m is not None
        assert m.groups() == ("mod", "", "", "func", "0", "iter")

    def test_embedded_in_stdout(self) -> None:
        s = "some output\n!$######mod:test_fn:f:1:x######$!\nmore output\n"
        m = matches_re_start.search(s)
        assert m is not None
        assert m.groups() == ("mod", "", "test_fn", "f", "1", "x")

    def test_multiple_matches(self) -> None:
        s = (
            "!$######m1:C1.fn1:t1:1:a######$!\n"
            "!$######m2:fn2:t2:2:b######$!\n"
        )
        matches = list(matches_re_start.finditer(s))
        assert len(matches) == 2
        assert matches[0].groups() == ("m1", "C1.", "fn1", "t1", "1", "a")
        assert matches[1].groups() == ("m2", "", "fn2", "t2", "2", "b")

    def test_no_match_without_newline(self) -> None:
        s = "!$######mod:test_fn:f:1:x######$!"
        m = matches_re_start.search(s)
        assert m is None

    def test_dots_in_module_path(self) -> None:
        s = "!$######a.b.c.d.e:test_fn:f:1:x######$!\n"
        m = matches_re_start.search(s)
        assert m is not None
        assert m.group(1) == "a.b.c.d.e"


# --- matches_re_end tests ---


class TestMatchesReEnd:
    def test_simple_no_class_with_runtime(self) -> None:
        s = "!######tests.test_foo:test_bar:target_func:1:abc:12345######!"
        m = matches_re_end.search(s)
        assert m is not None
        assert m.groups() == ("tests.test_foo", "", "test_bar", "target_func", "1", "abc:12345")

    def test_with_class_no_runtime(self) -> None:
        s = "!######tests.test_foo:MyClass.test_bar:target_func:1:abc######!"
        m = matches_re_end.search(s)
        assert m is not None
        assert m.groups() == ("tests.test_foo", "MyClass.", "test_bar", "target_func", "1", "abc")

    def test_nested_class_with_runtime(self) -> None:
        s = "!######mod:A.B.test_x:func:3:id123:99999######!"
        m = matches_re_end.search(s)
        assert m is not None
        assert m.groups() == ("mod", "A.B.", "test_x", "func", "3", "id123:99999")

    def test_runtime_colon_preserved_in_group6(self) -> None:
        """Group 6 must capture 'iteration_id:runtime' as a single string (colon included)."""
        s = "!######m:fn:f:1:iter42:98765######!"
        m = matches_re_end.search(s)
        assert m is not None
        assert m.group(6) == "iter42:98765"

    def test_embedded_in_stdout(self) -> None:
        s = "captured output\n!######mod:test_fn:f:1:x:500######!\nmore"
        m = matches_re_end.search(s)
        assert m is not None
        assert m.groups() == ("mod", "", "test_fn", "f", "1", "x:500")


# --- Start/End pairing (simulates parse_test_xml matching logic) ---


class TestStartEndPairing:
    def test_paired_markers(self) -> None:
        stdout = (
            "!$######mod:Class.test_fn:func:1:iter1######$!\n"
            "test output here\n"
            "!######mod:Class.test_fn:func:1:iter1:54321######!"
        )
        starts = list(matches_re_start.finditer(stdout))
        ends = {}
        for match in matches_re_end.finditer(stdout):
            groups = match.groups()
            g5 = groups[5]
            colon_pos = g5.find(":")
            if colon_pos != -1:
                key = groups[:5] + (g5[:colon_pos],)
            else:
                key = groups
            ends[key] = match

        assert len(starts) == 1
        assert len(ends) == 1
        # Start and end should pair on the first 5 groups + iteration_id
        start_groups = starts[0].groups()
        assert start_groups in ends


# --- parse_test_failures_from_stdout tests ---


class TestParseTestFailuresHeader:
    def test_standard_pytest_header(self) -> None:
        stdout = (
            "..F.\n"
            "=================================== FAILURES ===================================\n"
            "_______ test_foo _______\n"
            "\n"
            "    def test_foo():\n"
            ">       assert False\n"
            "E       AssertionError\n"
            "\n"
            "test.py:3: AssertionError\n"
            "=========================== short test summary info ============================\n"
            "FAILED test.py::test_foo\n"
        )
        result = parse_test_failures_from_stdout(stdout)
        assert "test_foo" in result

    def test_minimal_equals(self) -> None:
        """Even a short '= FAILURES =' header should be detected."""
        stdout = (
            "= FAILURES =\n"
            "_______ test_bar _______\n"
            "\n"
            "    assert False\n"
            "\n"
            "test.py:1: AssertionError\n"
            "= short test summary info =\n"
        )
        result = parse_test_failures_from_stdout(stdout)
        assert "test_bar" in result

    def test_no_failures_section(self) -> None:
        stdout = "....\n4 passed in 0.1s\n"
        result = parse_test_failures_from_stdout(stdout)
        assert result == {}

    def test_word_failures_without_equals_is_not_matched(self) -> None:
        """'FAILURES' without surrounding '=' signs should not trigger the header detection."""
        stdout = (
            "FAILURES detected in module\n"
            "_______ test_baz _______\n"
            "\n"
            "    assert False\n"
        )
        result = parse_test_failures_from_stdout(stdout)
        assert result == {}

    def test_failures_in_test_output_not_matched(self) -> None:
        """A test printing 'FAILURES' (no = signs) should not trigger header detection."""
        stdout = (
            "Testing FAILURES handling\n"
            "All good\n"
        )
        result = parse_test_failures_from_stdout(stdout)
        assert result == {}
