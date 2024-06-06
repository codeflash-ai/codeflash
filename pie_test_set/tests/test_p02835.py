from pie_test_set.p02835 import problem_p02835


def test_problem_p02835_0():
    actual_output = problem_p02835("5 7 9")
    expected_output = "win"
    assert str(actual_output) == expected_output


def test_problem_p02835_1():
    actual_output = problem_p02835("5 7 9")
    expected_output = "win"
    assert str(actual_output) == expected_output


def test_problem_p02835_2():
    actual_output = problem_p02835("13 7 2")
    expected_output = "bust"
    assert str(actual_output) == expected_output
