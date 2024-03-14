from code_to_optimize.pie_test_set.p03200 import problem_p03200


def test_problem_p03200_0():
    actual_output = problem_p03200("BBW")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03200_1():
    actual_output = problem_p03200("BWBWBW")
    expected_output = "6"
    assert str(actual_output) == expected_output


def test_problem_p03200_2():
    actual_output = problem_p03200("BBW")
    expected_output = "2"
    assert str(actual_output) == expected_output
