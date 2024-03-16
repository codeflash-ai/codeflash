from code_to_optimize.pie_test_set.p03146 import problem_p03146


def test_problem_p03146_0():
    actual_output = problem_p03146("8")
    expected_output = "5"
    assert str(actual_output) == expected_output


def test_problem_p03146_1():
    actual_output = problem_p03146("54")
    expected_output = "114"
    assert str(actual_output) == expected_output


def test_problem_p03146_2():
    actual_output = problem_p03146("8")
    expected_output = "5"
    assert str(actual_output) == expected_output


def test_problem_p03146_3():
    actual_output = problem_p03146("7")
    expected_output = "18"
    assert str(actual_output) == expected_output
