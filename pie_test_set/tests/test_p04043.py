from pie_test_set.p04043 import problem_p04043


def test_problem_p04043_0():
    actual_output = problem_p04043("5 5 7")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p04043_1():
    actual_output = problem_p04043("5 5 7")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p04043_2():
    actual_output = problem_p04043("7 7 5")
    expected_output = "NO"
    assert str(actual_output) == expected_output
