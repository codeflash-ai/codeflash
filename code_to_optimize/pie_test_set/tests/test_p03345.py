from code_to_optimize.pie_test_set.p03345 import problem_p03345


def test_problem_p03345_0():
    actual_output = problem_p03345("1 2 3 1")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03345_1():
    actual_output = problem_p03345("1 2 3 1")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03345_2():
    actual_output = problem_p03345("2 3 2 0")
    expected_output = "-1"
    assert str(actual_output) == expected_output


def test_problem_p03345_3():
    actual_output = problem_p03345("1000000000 1000000000 1000000000 1000000000000000000")
    expected_output = "0"
    assert str(actual_output) == expected_output
