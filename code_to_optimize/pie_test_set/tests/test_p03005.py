from code_to_optimize.pie_test_set.p03005 import problem_p03005


def test_problem_p03005_0():
    actual_output = problem_p03005("3 2")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03005_1():
    actual_output = problem_p03005("8 5")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03005_2():
    actual_output = problem_p03005("3 1")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03005_3():
    actual_output = problem_p03005("3 2")
    expected_output = "1"
    assert str(actual_output) == expected_output
