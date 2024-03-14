from code_to_optimize.pie_test_set.p03125 import problem_p03125


def test_problem_p03125_0():
    actual_output = problem_p03125("4 12")
    expected_output = "16"
    assert str(actual_output) == expected_output


def test_problem_p03125_1():
    actual_output = problem_p03125("4 12")
    expected_output = "16"
    assert str(actual_output) == expected_output


def test_problem_p03125_2():
    actual_output = problem_p03125("8 20")
    expected_output = "12"
    assert str(actual_output) == expected_output


def test_problem_p03125_3():
    actual_output = problem_p03125("1 1")
    expected_output = "2"
    assert str(actual_output) == expected_output
