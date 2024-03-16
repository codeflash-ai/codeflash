from code_to_optimize.pie_test_set.p02881 import problem_p02881


def test_problem_p02881_0():
    actual_output = problem_p02881("10")
    expected_output = "5"
    assert str(actual_output) == expected_output


def test_problem_p02881_1():
    actual_output = problem_p02881("10000000019")
    expected_output = "10000000018"
    assert str(actual_output) == expected_output


def test_problem_p02881_2():
    actual_output = problem_p02881("50")
    expected_output = "13"
    assert str(actual_output) == expected_output


def test_problem_p02881_3():
    actual_output = problem_p02881("10")
    expected_output = "5"
    assert str(actual_output) == expected_output
