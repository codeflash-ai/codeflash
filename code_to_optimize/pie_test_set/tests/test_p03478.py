from code_to_optimize.pie_test_set.p03478 import problem_p03478


def test_problem_p03478_0():
    actual_output = problem_p03478("20 2 5")
    expected_output = "84"
    assert str(actual_output) == expected_output


def test_problem_p03478_1():
    actual_output = problem_p03478("10 1 2")
    expected_output = "13"
    assert str(actual_output) == expected_output


def test_problem_p03478_2():
    actual_output = problem_p03478("20 2 5")
    expected_output = "84"
    assert str(actual_output) == expected_output


def test_problem_p03478_3():
    actual_output = problem_p03478("100 4 16")
    expected_output = "4554"
    assert str(actual_output) == expected_output
