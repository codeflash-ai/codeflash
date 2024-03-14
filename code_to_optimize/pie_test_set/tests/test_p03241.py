from code_to_optimize.pie_test_set.p03241 import problem_p03241


def test_problem_p03241_0():
    actual_output = problem_p03241("3 14")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03241_1():
    actual_output = problem_p03241("10 123")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03241_2():
    actual_output = problem_p03241("3 14")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03241_3():
    actual_output = problem_p03241("100000 1000000000")
    expected_output = "10000"
    assert str(actual_output) == expected_output
