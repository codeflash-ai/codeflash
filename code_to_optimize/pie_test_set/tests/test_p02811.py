from code_to_optimize.pie_test_set.p02811 import problem_p02811


def test_problem_p02811_0():
    actual_output = problem_p02811("2 900")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02811_1():
    actual_output = problem_p02811("1 501")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p02811_2():
    actual_output = problem_p02811("4 2000")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02811_3():
    actual_output = problem_p02811("2 900")
    expected_output = "Yes"
    assert str(actual_output) == expected_output
