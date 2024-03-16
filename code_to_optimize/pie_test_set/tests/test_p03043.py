from code_to_optimize.pie_test_set.p03043 import problem_p03043


def test_problem_p03043_0():
    actual_output = problem_p03043("3 10")
    expected_output = "0.145833333333"
    assert str(actual_output) == expected_output


def test_problem_p03043_1():
    actual_output = problem_p03043("3 10")
    expected_output = "0.145833333333"
    assert str(actual_output) == expected_output


def test_problem_p03043_2():
    actual_output = problem_p03043("100000 5")
    expected_output = "0.999973749998"
    assert str(actual_output) == expected_output
