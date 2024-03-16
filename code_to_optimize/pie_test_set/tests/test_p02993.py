from code_to_optimize.pie_test_set.p02993 import problem_p02993


def test_problem_p02993_0():
    actual_output = problem_p02993("3776")
    expected_output = "Bad"
    assert str(actual_output) == expected_output


def test_problem_p02993_1():
    actual_output = problem_p02993("1333")
    expected_output = "Bad"
    assert str(actual_output) == expected_output


def test_problem_p02993_2():
    actual_output = problem_p02993("8080")
    expected_output = "Good"
    assert str(actual_output) == expected_output


def test_problem_p02993_3():
    actual_output = problem_p02993("3776")
    expected_output = "Bad"
    assert str(actual_output) == expected_output


def test_problem_p02993_4():
    actual_output = problem_p02993("0024")
    expected_output = "Bad"
    assert str(actual_output) == expected_output
