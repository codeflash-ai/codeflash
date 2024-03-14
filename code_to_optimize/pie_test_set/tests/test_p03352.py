from code_to_optimize.pie_test_set.p03352 import problem_p03352


def test_problem_p03352_0():
    actual_output = problem_p03352("10")
    expected_output = "9"
    assert str(actual_output) == expected_output


def test_problem_p03352_1():
    actual_output = problem_p03352("10")
    expected_output = "9"
    assert str(actual_output) == expected_output


def test_problem_p03352_2():
    actual_output = problem_p03352("1")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03352_3():
    actual_output = problem_p03352("999")
    expected_output = "961"
    assert str(actual_output) == expected_output
