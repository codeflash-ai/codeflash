from pie_test_set.p03838 import problem_p03838


def test_problem_p03838_0():
    actual_output = problem_p03838("10 20")
    expected_output = "10"
    assert str(actual_output) == expected_output


def test_problem_p03838_1():
    actual_output = problem_p03838("10 20")
    expected_output = "10"
    assert str(actual_output) == expected_output


def test_problem_p03838_2():
    actual_output = problem_p03838("-10 -20")
    expected_output = "12"
    assert str(actual_output) == expected_output


def test_problem_p03838_3():
    actual_output = problem_p03838("10 -10")
    expected_output = "1"
    assert str(actual_output) == expected_output
