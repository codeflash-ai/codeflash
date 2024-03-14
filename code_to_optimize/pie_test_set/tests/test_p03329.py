from code_to_optimize.pie_test_set.p03329 import problem_p03329


def test_problem_p03329_0():
    actual_output = problem_p03329("127")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03329_1():
    actual_output = problem_p03329("3")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03329_2():
    actual_output = problem_p03329("44852")
    expected_output = "16"
    assert str(actual_output) == expected_output


def test_problem_p03329_3():
    actual_output = problem_p03329("127")
    expected_output = "4"
    assert str(actual_output) == expected_output
