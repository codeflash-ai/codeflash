from code_to_optimize.pie_test_set.p03609 import problem_p03609


def test_problem_p03609_0():
    actual_output = problem_p03609("100 17")
    expected_output = "83"
    assert str(actual_output) == expected_output


def test_problem_p03609_1():
    actual_output = problem_p03609("48 58")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03609_2():
    actual_output = problem_p03609("1000000000 1000000000")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03609_3():
    actual_output = problem_p03609("100 17")
    expected_output = "83"
    assert str(actual_output) == expected_output
