from pie_test_set.p03671 import problem_p03671


def test_problem_p03671_0():
    actual_output = problem_p03671("700 600 780")
    expected_output = "1300"
    assert str(actual_output) == expected_output


def test_problem_p03671_1():
    actual_output = problem_p03671("10000 10000 10000")
    expected_output = "20000"
    assert str(actual_output) == expected_output


def test_problem_p03671_2():
    actual_output = problem_p03671("700 600 780")
    expected_output = "1300"
    assert str(actual_output) == expected_output
