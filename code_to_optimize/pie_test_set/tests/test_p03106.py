from code_to_optimize.pie_test_set.p03106 import problem_p03106


def test_problem_p03106_0():
    actual_output = problem_p03106("8 12 2")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03106_1():
    actual_output = problem_p03106("1 1 1")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03106_2():
    actual_output = problem_p03106("100 50 4")
    expected_output = "5"
    assert str(actual_output) == expected_output


def test_problem_p03106_3():
    actual_output = problem_p03106("8 12 2")
    expected_output = "2"
    assert str(actual_output) == expected_output
