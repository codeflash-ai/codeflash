from code_to_optimize.pie_test_set.p03047 import problem_p03047


def test_problem_p03047_0():
    actual_output = problem_p03047("3 2")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03047_1():
    actual_output = problem_p03047("13 3")
    expected_output = "11"
    assert str(actual_output) == expected_output


def test_problem_p03047_2():
    actual_output = problem_p03047("3 2")
    expected_output = "2"
    assert str(actual_output) == expected_output
