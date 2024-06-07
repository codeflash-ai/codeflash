from pie_test_set.p02471 import problem_p02471


def test_problem_p02471_0():
    actual_output = problem_p02471("4 12")
    expected_output = "1 0"
    assert str(actual_output) == expected_output


def test_problem_p02471_1():
    actual_output = problem_p02471("3 8")
    expected_output = "3 -1"
    assert str(actual_output) == expected_output


def test_problem_p02471_2():
    actual_output = problem_p02471("4 12")
    expected_output = "1 0"
    assert str(actual_output) == expected_output
