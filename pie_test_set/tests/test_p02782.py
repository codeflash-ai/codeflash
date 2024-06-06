from pie_test_set.p02782 import problem_p02782


def test_problem_p02782_0():
    actual_output = problem_p02782("1 1 2 2")
    expected_output = "14"
    assert str(actual_output) == expected_output


def test_problem_p02782_1():
    actual_output = problem_p02782("314 159 2653 589")
    expected_output = "602215194"
    assert str(actual_output) == expected_output


def test_problem_p02782_2():
    actual_output = problem_p02782("1 1 2 2")
    expected_output = "14"
    assert str(actual_output) == expected_output
