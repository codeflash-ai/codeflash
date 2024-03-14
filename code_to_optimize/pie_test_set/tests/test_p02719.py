from code_to_optimize.pie_test_set.p02719 import problem_p02719


def test_problem_p02719_0():
    actual_output = problem_p02719("7 4")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p02719_1():
    actual_output = problem_p02719("1000000000000000000 1")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p02719_2():
    actual_output = problem_p02719("7 4")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p02719_3():
    actual_output = problem_p02719("2 6")
    expected_output = "2"
    assert str(actual_output) == expected_output
