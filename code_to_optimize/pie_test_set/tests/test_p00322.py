from code_to_optimize.pie_test_set.p00322 import problem_p00322


def test_problem_p00322_0():
    actual_output = problem_p00322("7 6 -1 1 -1 9 2 3 4")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p00322_1():
    actual_output = problem_p00322("7 6 5 1 8 9 2 3 4")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p00322_2():
    actual_output = problem_p00322("-1 -1 -1 -1 -1 -1 8 4 6")
    expected_output = "12"
    assert str(actual_output) == expected_output


def test_problem_p00322_3():
    actual_output = problem_p00322("7 6 -1 1 -1 9 2 3 4")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p00322_4():
    actual_output = problem_p00322("-1 -1 -1 -1 -1 -1 -1 -1 -1")
    expected_output = "168"
    assert str(actual_output) == expected_output
