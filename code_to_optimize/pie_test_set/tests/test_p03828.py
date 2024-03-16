from code_to_optimize.pie_test_set.p03828 import problem_p03828


def test_problem_p03828_0():
    actual_output = problem_p03828("3")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03828_1():
    actual_output = problem_p03828("6")
    expected_output = "30"
    assert str(actual_output) == expected_output


def test_problem_p03828_2():
    actual_output = problem_p03828("3")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03828_3():
    actual_output = problem_p03828("1000")
    expected_output = "972926972"
    assert str(actual_output) == expected_output
