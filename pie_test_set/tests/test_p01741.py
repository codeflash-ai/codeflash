from pie_test_set.p01741 import problem_p01741


def test_problem_p01741_0():
    actual_output = problem_p01741("1.000")
    expected_output = "2.000000000000"
    assert str(actual_output) == expected_output


def test_problem_p01741_1():
    actual_output = problem_p01741("1.000")
    expected_output = "2.000000000000"
    assert str(actual_output) == expected_output


def test_problem_p01741_2():
    actual_output = problem_p01741("2.345")
    expected_output = "3.316330803765"
    assert str(actual_output) == expected_output
