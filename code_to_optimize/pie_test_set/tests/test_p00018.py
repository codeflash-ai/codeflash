from code_to_optimize.pie_test_set.p00018 import problem_p00018


def test_problem_p00018_0():
    actual_output = problem_p00018("3 6 9 7 5")
    expected_output = "9 7 6 5 3"
    assert str(actual_output) == expected_output
