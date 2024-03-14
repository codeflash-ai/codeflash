from code_to_optimize.pie_test_set.p03632 import problem_p03632


def test_problem_p03632_0():
    actual_output = problem_p03632("0 75 25 100")
    expected_output = "50"
    assert str(actual_output) == expected_output


def test_problem_p03632_1():
    actual_output = problem_p03632("10 90 20 80")
    expected_output = "60"
    assert str(actual_output) == expected_output


def test_problem_p03632_2():
    actual_output = problem_p03632("0 75 25 100")
    expected_output = "50"
    assert str(actual_output) == expected_output


def test_problem_p03632_3():
    actual_output = problem_p03632("0 33 66 99")
    expected_output = "0"
    assert str(actual_output) == expected_output
