from code_to_optimize.pie_test_set.p02554 import problem_p02554


def test_problem_p02554_0():
    actual_output = problem_p02554("2")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p02554_1():
    actual_output = problem_p02554("869121")
    expected_output = "2511445"
    assert str(actual_output) == expected_output


def test_problem_p02554_2():
    actual_output = problem_p02554("1")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p02554_3():
    actual_output = problem_p02554("2")
    expected_output = "2"
    assert str(actual_output) == expected_output
