from code_to_optimize.pie_test_set.p02818 import problem_p02818


def test_problem_p02818_0():
    actual_output = problem_p02818("2 3 3")
    expected_output = "0 2"
    assert str(actual_output) == expected_output


def test_problem_p02818_1():
    actual_output = problem_p02818("500000000000 500000000000 1000000000000")
    expected_output = "0 0"
    assert str(actual_output) == expected_output


def test_problem_p02818_2():
    actual_output = problem_p02818("2 3 3")
    expected_output = "0 2"
    assert str(actual_output) == expected_output
