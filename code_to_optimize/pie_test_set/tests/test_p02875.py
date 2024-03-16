from code_to_optimize.pie_test_set.p02875 import problem_p02875


def test_problem_p02875_0():
    actual_output = problem_p02875("2")
    expected_output = "7"
    assert str(actual_output) == expected_output


def test_problem_p02875_1():
    actual_output = problem_p02875("2")
    expected_output = "7"
    assert str(actual_output) == expected_output


def test_problem_p02875_2():
    actual_output = problem_p02875("1000000")
    expected_output = "210055358"
    assert str(actual_output) == expected_output


def test_problem_p02875_3():
    actual_output = problem_p02875("10")
    expected_output = "50007"
    assert str(actual_output) == expected_output
