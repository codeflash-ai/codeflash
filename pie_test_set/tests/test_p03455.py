from pie_test_set.p03455 import problem_p03455


def test_problem_p03455_0():
    actual_output = problem_p03455("3 4")
    expected_output = "Even"
    assert str(actual_output) == expected_output


def test_problem_p03455_1():
    actual_output = problem_p03455("3 4")
    expected_output = "Even"
    assert str(actual_output) == expected_output


def test_problem_p03455_2():
    actual_output = problem_p03455("1 21")
    expected_output = "Odd"
    assert str(actual_output) == expected_output
