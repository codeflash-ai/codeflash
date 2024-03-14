from code_to_optimize.pie_test_set.p02939 import problem_p02939


def test_problem_p02939_0():
    actual_output = problem_p02939("aabbaa")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p02939_1():
    actual_output = problem_p02939("aaaccacabaababc")
    expected_output = "12"
    assert str(actual_output) == expected_output


def test_problem_p02939_2():
    actual_output = problem_p02939("aabbaa")
    expected_output = "4"
    assert str(actual_output) == expected_output
