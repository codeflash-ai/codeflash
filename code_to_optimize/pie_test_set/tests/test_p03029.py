from code_to_optimize.pie_test_set.p03029 import problem_p03029


def test_problem_p03029_0():
    actual_output = problem_p03029("1 3")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03029_1():
    actual_output = problem_p03029("1 3")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03029_2():
    actual_output = problem_p03029("32 21")
    expected_output = "58"
    assert str(actual_output) == expected_output


def test_problem_p03029_3():
    actual_output = problem_p03029("0 1")
    expected_output = "0"
    assert str(actual_output) == expected_output
