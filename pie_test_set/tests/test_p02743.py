from pie_test_set.p02743 import problem_p02743


def test_problem_p02743_0():
    actual_output = problem_p02743("2 3 9")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p02743_1():
    actual_output = problem_p02743("2 3 10")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02743_2():
    actual_output = problem_p02743("2 3 9")
    expected_output = "No"
    assert str(actual_output) == expected_output
