from code_to_optimize.pie_test_set.p02659 import problem_p02659


def test_problem_p02659_0():
    actual_output = problem_p02659("198 1.10")
    expected_output = "217"
    assert str(actual_output) == expected_output


def test_problem_p02659_1():
    actual_output = problem_p02659("1000000000000000 9.99")
    expected_output = "9990000000000000"
    assert str(actual_output) == expected_output


def test_problem_p02659_2():
    actual_output = problem_p02659("1 0.01")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p02659_3():
    actual_output = problem_p02659("198 1.10")
    expected_output = "217"
    assert str(actual_output) == expected_output
