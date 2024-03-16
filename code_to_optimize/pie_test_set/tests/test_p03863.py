from code_to_optimize.pie_test_set.p03863 import problem_p03863


def test_problem_p03863_0():
    actual_output = problem_p03863("aba")
    expected_output = "Second"
    assert str(actual_output) == expected_output


def test_problem_p03863_1():
    actual_output = problem_p03863("abc")
    expected_output = "First"
    assert str(actual_output) == expected_output


def test_problem_p03863_2():
    actual_output = problem_p03863("aba")
    expected_output = "Second"
    assert str(actual_output) == expected_output


def test_problem_p03863_3():
    actual_output = problem_p03863("abcab")
    expected_output = "First"
    assert str(actual_output) == expected_output
