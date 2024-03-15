from code_to_optimize.pie_test_set.p04004 import problem_p04004


def test_problem_p04004_0():
    actual_output = problem_p04004("1 1 1")
    expected_output = "17"
    assert str(actual_output) == expected_output


def test_problem_p04004_1():
    actual_output = problem_p04004("4 2 2")
    expected_output = "1227"
    assert str(actual_output) == expected_output


def test_problem_p04004_2():
    actual_output = problem_p04004("1 1 1")
    expected_output = "17"
    assert str(actual_output) == expected_output


def test_problem_p04004_3():
    actual_output = problem_p04004("1000 1000 1000")
    expected_output = "261790852"
    assert str(actual_output) == expected_output
