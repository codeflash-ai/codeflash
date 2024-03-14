from code_to_optimize.pie_test_set.p02653 import problem_p02653


def test_problem_p02653_0():
    actual_output = problem_p02653("4 2 3")
    expected_output = "11"
    assert str(actual_output) == expected_output


def test_problem_p02653_1():
    actual_output = problem_p02653("4 2 3")
    expected_output = "11"
    assert str(actual_output) == expected_output


def test_problem_p02653_2():
    actual_output = problem_p02653("10 7 2")
    expected_output = "533"
    assert str(actual_output) == expected_output


def test_problem_p02653_3():
    actual_output = problem_p02653("1000 100 10")
    expected_output = "828178524"
    assert str(actual_output) == expected_output
