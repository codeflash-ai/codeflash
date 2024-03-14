from code_to_optimize.pie_test_set.p02553 import problem_p02553


def test_problem_p02553_0():
    actual_output = problem_p02553("1 2 1 1")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p02553_1():
    actual_output = problem_p02553("-1000000000 0 -1000000000 0")
    expected_output = "1000000000000000000"
    assert str(actual_output) == expected_output


def test_problem_p02553_2():
    actual_output = problem_p02553("1 2 1 1")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p02553_3():
    actual_output = problem_p02553("3 5 -4 -2")
    expected_output = "-6"
    assert str(actual_output) == expected_output
