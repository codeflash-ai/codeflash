from code_to_optimize.pie_test_set.p02927 import problem_p02927


def test_problem_p02927_0():
    actual_output = problem_p02927("15 40")
    expected_output = "10"
    assert str(actual_output) == expected_output


def test_problem_p02927_1():
    actual_output = problem_p02927("15 40")
    expected_output = "10"
    assert str(actual_output) == expected_output


def test_problem_p02927_2():
    actual_output = problem_p02927("1 1")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p02927_3():
    actual_output = problem_p02927("12 31")
    expected_output = "5"
    assert str(actual_output) == expected_output
