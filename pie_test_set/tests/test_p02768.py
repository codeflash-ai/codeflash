from pie_test_set.p02768 import problem_p02768


def test_problem_p02768_0():
    actual_output = problem_p02768("4 1 3")
    expected_output = "7"
    assert str(actual_output) == expected_output


def test_problem_p02768_1():
    actual_output = problem_p02768("4 1 3")
    expected_output = "7"
    assert str(actual_output) == expected_output


def test_problem_p02768_2():
    actual_output = problem_p02768("1000000000 141421 173205")
    expected_output = "34076506"
    assert str(actual_output) == expected_output
