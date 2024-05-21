from pie_test_set.p03826 import problem_p03826


def test_problem_p03826_0():
    actual_output = problem_p03826("3 5 2 7")
    expected_output = "15"
    assert str(actual_output) == expected_output


def test_problem_p03826_1():
    actual_output = problem_p03826("100 600 200 300")
    expected_output = "60000"
    assert str(actual_output) == expected_output


def test_problem_p03826_2():
    actual_output = problem_p03826("3 5 2 7")
    expected_output = "15"
    assert str(actual_output) == expected_output
