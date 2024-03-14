from code_to_optimize.pie_test_set.p03135 import problem_p03135


def test_problem_p03135_0():
    actual_output = problem_p03135("8 3")
    expected_output = "2.6666666667"
    assert str(actual_output) == expected_output


def test_problem_p03135_1():
    actual_output = problem_p03135("1 100")
    expected_output = "0.0100000000"
    assert str(actual_output) == expected_output


def test_problem_p03135_2():
    actual_output = problem_p03135("99 1")
    expected_output = "99.0000000000"
    assert str(actual_output) == expected_output


def test_problem_p03135_3():
    actual_output = problem_p03135("8 3")
    expected_output = "2.6666666667"
    assert str(actual_output) == expected_output
