from code_to_optimize.pie_test_set.p03286 import problem_p03286


def test_problem_p03286_0():
    actual_output = problem_p03286("-9")
    expected_output = "1011"
    assert str(actual_output) == expected_output


def test_problem_p03286_1():
    actual_output = problem_p03286("0")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03286_2():
    actual_output = problem_p03286("123456789")
    expected_output = "11000101011001101110100010101"
    assert str(actual_output) == expected_output


def test_problem_p03286_3():
    actual_output = problem_p03286("-9")
    expected_output = "1011"
    assert str(actual_output) == expected_output
