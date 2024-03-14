from code_to_optimize.pie_test_set.p03501 import problem_p03501


def test_problem_p03501_0():
    actual_output = problem_p03501("7 17 120")
    expected_output = "119"
    assert str(actual_output) == expected_output
