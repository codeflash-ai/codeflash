from code_to_optimize.pie_test_set.p00341 import problem_p00341


def test_problem_p00341_0():
    actual_output = problem_p00341("1 1 3 4 8 9 7 3 4 5 5 5")
    expected_output = "no"
    assert str(actual_output) == expected_output


def test_problem_p00341_1():
    actual_output = problem_p00341("1 1 2 2 3 1 2 3 3 3 1 2")
    expected_output = "yes"
    assert str(actual_output) == expected_output


def test_problem_p00341_2():
    actual_output = problem_p00341("1 1 3 4 8 9 7 3 4 5 5 5")
    expected_output = "no"
    assert str(actual_output) == expected_output
