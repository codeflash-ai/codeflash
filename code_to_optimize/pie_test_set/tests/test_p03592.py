from code_to_optimize.pie_test_set.p03592 import problem_p03592


def test_problem_p03592_0():
    actual_output = problem_p03592("2 2 2")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03592_1():
    actual_output = problem_p03592("2 2 2")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03592_2():
    actual_output = problem_p03592("2 2 1")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p03592_3():
    actual_output = problem_p03592("7 9 20")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p03592_4():
    actual_output = problem_p03592("3 5 8")
    expected_output = "Yes"
    assert str(actual_output) == expected_output
