from code_to_optimize.pie_test_set.p03035 import problem_p03035


def test_problem_p03035_0():
    actual_output = problem_p03035("30 100")
    expected_output = "100"
    assert str(actual_output) == expected_output


def test_problem_p03035_1():
    actual_output = problem_p03035("0 100")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03035_2():
    actual_output = problem_p03035("30 100")
    expected_output = "100"
    assert str(actual_output) == expected_output


def test_problem_p03035_3():
    actual_output = problem_p03035("12 100")
    expected_output = "50"
    assert str(actual_output) == expected_output
