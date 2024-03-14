from code_to_optimize.pie_test_set.p02470 import problem_p02470


def test_problem_p02470_0():
    actual_output = problem_p02470("6")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p02470_1():
    actual_output = problem_p02470("1000000")
    expected_output = "400000"
    assert str(actual_output) == expected_output


def test_problem_p02470_2():
    actual_output = problem_p02470("6")
    expected_output = "2"
    assert str(actual_output) == expected_output
