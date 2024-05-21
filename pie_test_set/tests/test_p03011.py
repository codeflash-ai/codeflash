from pie_test_set.p03011 import problem_p03011


def test_problem_p03011_0():
    actual_output = problem_p03011("1 3 4")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03011_1():
    actual_output = problem_p03011("3 2 3")
    expected_output = "5"
    assert str(actual_output) == expected_output


def test_problem_p03011_2():
    actual_output = problem_p03011("1 3 4")
    expected_output = "4"
    assert str(actual_output) == expected_output
