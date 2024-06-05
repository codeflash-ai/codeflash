from pie_test_set.p02468 import problem_p02468


def test_problem_p02468_0():
    actual_output = problem_p02468("2 3")
    expected_output = "8"
    assert str(actual_output) == expected_output


def test_problem_p02468_1():
    actual_output = problem_p02468("2 3")
    expected_output = "8"
    assert str(actual_output) == expected_output


def test_problem_p02468_2():
    actual_output = problem_p02468("5 8")
    expected_output = "390625"
    assert str(actual_output) == expected_output
