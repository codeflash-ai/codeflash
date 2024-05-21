from pie_test_set.p03253 import problem_p03253


def test_problem_p03253_0():
    actual_output = problem_p03253("2 6")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03253_1():
    actual_output = problem_p03253("3 12")
    expected_output = "18"
    assert str(actual_output) == expected_output


def test_problem_p03253_2():
    actual_output = problem_p03253("2 6")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03253_3():
    actual_output = problem_p03253("100000 1000000000")
    expected_output = "957870001"
    assert str(actual_output) == expected_output
