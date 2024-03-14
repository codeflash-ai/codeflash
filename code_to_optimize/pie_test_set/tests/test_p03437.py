from code_to_optimize.pie_test_set.p03437 import problem_p03437


def test_problem_p03437_0():
    actual_output = problem_p03437("8 6")
    expected_output = "16"
    assert str(actual_output) == expected_output


def test_problem_p03437_1():
    actual_output = problem_p03437("3 3")
    expected_output = "-1"
    assert str(actual_output) == expected_output


def test_problem_p03437_2():
    actual_output = problem_p03437("8 6")
    expected_output = "16"
    assert str(actual_output) == expected_output
