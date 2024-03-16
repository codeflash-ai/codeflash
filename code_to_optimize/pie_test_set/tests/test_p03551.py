from code_to_optimize.pie_test_set.p03551 import problem_p03551


def test_problem_p03551_0():
    actual_output = problem_p03551("1 1")
    expected_output = "3800"
    assert str(actual_output) == expected_output
