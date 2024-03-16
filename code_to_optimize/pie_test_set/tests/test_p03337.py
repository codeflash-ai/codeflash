from code_to_optimize.pie_test_set.p03337 import problem_p03337


def test_problem_p03337_0():
    actual_output = problem_p03337("3 1")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03337_1():
    actual_output = problem_p03337("0 0")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03337_2():
    actual_output = problem_p03337("3 1")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03337_3():
    actual_output = problem_p03337("4 -2")
    expected_output = "6"
    assert str(actual_output) == expected_output
