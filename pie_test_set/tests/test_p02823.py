from pie_test_set.p02823 import problem_p02823


def test_problem_p02823_0():
    actual_output = problem_p02823("5 2 4")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p02823_1():
    actual_output = problem_p02823("5 2 3")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p02823_2():
    actual_output = problem_p02823("5 2 4")
    expected_output = "1"
    assert str(actual_output) == expected_output
