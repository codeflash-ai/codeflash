from pie_test_set.p03623 import problem_p03623


def test_problem_p03623_0():
    actual_output = problem_p03623("5 2 7")
    expected_output = "B"
    assert str(actual_output) == expected_output


def test_problem_p03623_1():
    actual_output = problem_p03623("1 999 1000")
    expected_output = "A"
    assert str(actual_output) == expected_output


def test_problem_p03623_2():
    actual_output = problem_p03623("5 2 7")
    expected_output = "B"
    assert str(actual_output) == expected_output
