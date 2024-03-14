from code_to_optimize.pie_test_set.p02992 import problem_p02992


def test_problem_p02992_0():
    actual_output = problem_p02992("3 2")
    expected_output = "5"
    assert str(actual_output) == expected_output


def test_problem_p02992_1():
    actual_output = problem_p02992("10 3")
    expected_output = "147"
    assert str(actual_output) == expected_output


def test_problem_p02992_2():
    actual_output = problem_p02992("314159265 35")
    expected_output = "457397712"
    assert str(actual_output) == expected_output


def test_problem_p02992_3():
    actual_output = problem_p02992("3 2")
    expected_output = "5"
    assert str(actual_output) == expected_output
