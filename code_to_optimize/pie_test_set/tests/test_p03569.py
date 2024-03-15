from code_to_optimize.pie_test_set.p03569 import problem_p03569


def test_problem_p03569_0():
    actual_output = problem_p03569("xabxa")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03569_1():
    actual_output = problem_p03569("a")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03569_2():
    actual_output = problem_p03569("xabxa")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03569_3():
    actual_output = problem_p03569("ab")
    expected_output = "-1"
    assert str(actual_output) == expected_output


def test_problem_p03569_4():
    actual_output = problem_p03569("oxxx")
    expected_output = "3"
    assert str(actual_output) == expected_output
