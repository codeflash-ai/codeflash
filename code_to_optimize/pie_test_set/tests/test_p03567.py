from code_to_optimize.pie_test_set.p03567 import problem_p03567


def test_problem_p03567_0():
    actual_output = problem_p03567("BACD")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03567_1():
    actual_output = problem_p03567("CABD")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p03567_2():
    actual_output = problem_p03567("BACD")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03567_3():
    actual_output = problem_p03567("ACACA")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03567_4():
    actual_output = problem_p03567("ABCD")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p03567_5():
    actual_output = problem_p03567("XX")
    expected_output = "No"
    assert str(actual_output) == expected_output
