from code_to_optimize.pie_test_set.p01447 import problem_p01447


def test_problem_p01447_0():
    actual_output = problem_p01447("8")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p01447_1():
    actual_output = problem_p01447("30")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p01447_2():
    actual_output = problem_p01447("8")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p01447_3():
    actual_output = problem_p01447("2000000000")
    expected_output = "20"
    assert str(actual_output) == expected_output
