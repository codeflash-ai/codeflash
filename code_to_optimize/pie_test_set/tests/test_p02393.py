from code_to_optimize.pie_test_set.p02393 import problem_p02393


def test_problem_p02393_0():
    actual_output = problem_p02393("3 8 1")
    expected_output = "1 3 8"
    assert str(actual_output) == expected_output
