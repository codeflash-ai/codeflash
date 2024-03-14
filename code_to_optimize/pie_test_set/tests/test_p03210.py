from code_to_optimize.pie_test_set.p03210 import problem_p03210


def test_problem_p03210_0():
    actual_output = problem_p03210("5")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03210_1():
    actual_output = problem_p03210("6")
    expected_output = "NO"
    assert str(actual_output) == expected_output


def test_problem_p03210_2():
    actual_output = problem_p03210("5")
    expected_output = "YES"
    assert str(actual_output) == expected_output
