from pie_test_set.p02394 import problem_p02394


def test_problem_p02394_0():
    actual_output = problem_p02394("5 4 2 2 1")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02394_1():
    actual_output = problem_p02394("5 4 2 2 1")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02394_2():
    actual_output = problem_p02394("5 4 2 4 1")
    expected_output = "No"
    assert str(actual_output) == expected_output
