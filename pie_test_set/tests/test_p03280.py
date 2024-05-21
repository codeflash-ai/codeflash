from pie_test_set.p03280 import problem_p03280


def test_problem_p03280_0():
    actual_output = problem_p03280("2 2")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03280_1():
    actual_output = problem_p03280("5 7")
    expected_output = "24"
    assert str(actual_output) == expected_output


def test_problem_p03280_2():
    actual_output = problem_p03280("2 2")
    expected_output = "1"
    assert str(actual_output) == expected_output
