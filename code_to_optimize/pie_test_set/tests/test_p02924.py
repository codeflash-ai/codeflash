from code_to_optimize.pie_test_set.p02924 import problem_p02924


def test_problem_p02924_0():
    actual_output = problem_p02924("2")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p02924_1():
    actual_output = problem_p02924("13")
    expected_output = "78"
    assert str(actual_output) == expected_output


def test_problem_p02924_2():
    actual_output = problem_p02924("2")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p02924_3():
    actual_output = problem_p02924("1")
    expected_output = "0"
    assert str(actual_output) == expected_output
