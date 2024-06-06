from pie_test_set.p00340 import problem_p00340


def test_problem_p00340_0():
    actual_output = problem_p00340("1 1 3 4")
    expected_output = "no"
    assert str(actual_output) == expected_output


def test_problem_p00340_1():
    actual_output = problem_p00340("1 1 2 2")
    expected_output = "yes"
    assert str(actual_output) == expected_output


def test_problem_p00340_2():
    actual_output = problem_p00340("4 4 4 10")
    expected_output = "no"
    assert str(actual_output) == expected_output


def test_problem_p00340_3():
    actual_output = problem_p00340("2 1 1 2")
    expected_output = "yes"
    assert str(actual_output) == expected_output


def test_problem_p00340_4():
    actual_output = problem_p00340("1 1 3 4")
    expected_output = "no"
    assert str(actual_output) == expected_output
