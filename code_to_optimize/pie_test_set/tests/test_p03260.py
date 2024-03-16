from code_to_optimize.pie_test_set.p03260 import problem_p03260


def test_problem_p03260_0():
    actual_output = problem_p03260("3 1")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03260_1():
    actual_output = problem_p03260("3 1")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03260_2():
    actual_output = problem_p03260("1 2")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p03260_3():
    actual_output = problem_p03260("2 2")
    expected_output = "No"
    assert str(actual_output) == expected_output
