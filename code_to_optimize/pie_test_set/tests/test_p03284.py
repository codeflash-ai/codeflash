from code_to_optimize.pie_test_set.p03284 import problem_p03284


def test_problem_p03284_0():
    actual_output = problem_p03284("7 3")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03284_1():
    actual_output = problem_p03284("100 10")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03284_2():
    actual_output = problem_p03284("1 1")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03284_3():
    actual_output = problem_p03284("7 3")
    expected_output = "1"
    assert str(actual_output) == expected_output
