from code_to_optimize.pie_test_set.p02831 import problem_p02831


def test_problem_p02831_0():
    actual_output = problem_p02831("2 3")
    expected_output = "6"
    assert str(actual_output) == expected_output


def test_problem_p02831_1():
    actual_output = problem_p02831("100000 99999")
    expected_output = "9999900000"
    assert str(actual_output) == expected_output


def test_problem_p02831_2():
    actual_output = problem_p02831("123 456")
    expected_output = "18696"
    assert str(actual_output) == expected_output


def test_problem_p02831_3():
    actual_output = problem_p02831("2 3")
    expected_output = "6"
    assert str(actual_output) == expected_output
