from pie_test_set.p02974 import problem_p02974


def test_problem_p02974_0():
    actual_output = problem_p02974("3 2")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p02974_1():
    actual_output = problem_p02974("39 14")
    expected_output = "74764168"
    assert str(actual_output) == expected_output


def test_problem_p02974_2():
    actual_output = problem_p02974("3 2")
    expected_output = "2"
    assert str(actual_output) == expected_output
