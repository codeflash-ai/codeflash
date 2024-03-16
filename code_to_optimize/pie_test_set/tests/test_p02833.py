from code_to_optimize.pie_test_set.p02833 import problem_p02833


def test_problem_p02833_0():
    actual_output = problem_p02833("12")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p02833_1():
    actual_output = problem_p02833("1000000000000000000")
    expected_output = "124999999999999995"
    assert str(actual_output) == expected_output


def test_problem_p02833_2():
    actual_output = problem_p02833("5")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p02833_3():
    actual_output = problem_p02833("12")
    expected_output = "1"
    assert str(actual_output) == expected_output
