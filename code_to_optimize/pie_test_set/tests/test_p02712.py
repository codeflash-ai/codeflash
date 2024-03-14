from code_to_optimize.pie_test_set.p02712 import problem_p02712


def test_problem_p02712_0():
    actual_output = problem_p02712("15")
    expected_output = "60"
    assert str(actual_output) == expected_output


def test_problem_p02712_1():
    actual_output = problem_p02712("1000000")
    expected_output = "266666333332"
    assert str(actual_output) == expected_output


def test_problem_p02712_2():
    actual_output = problem_p02712("15")
    expected_output = "60"
    assert str(actual_output) == expected_output
