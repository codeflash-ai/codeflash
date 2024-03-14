from code_to_optimize.pie_test_set.p03840 import problem_p03840


def test_problem_p03840_0():
    actual_output = problem_p03840("2 1 1 0 0 0 0")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03840_1():
    actual_output = problem_p03840("2 1 1 0 0 0 0")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03840_2():
    actual_output = problem_p03840("0 0 10 0 0 0 0")
    expected_output = "0"
    assert str(actual_output) == expected_output
