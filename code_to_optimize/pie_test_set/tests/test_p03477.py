from code_to_optimize.pie_test_set.p03477 import problem_p03477


def test_problem_p03477_0():
    actual_output = problem_p03477("3 8 7 1")
    expected_output = "Left"
    assert str(actual_output) == expected_output


def test_problem_p03477_1():
    actual_output = problem_p03477("3 8 7 1")
    expected_output = "Left"
    assert str(actual_output) == expected_output


def test_problem_p03477_2():
    actual_output = problem_p03477("1 7 6 4")
    expected_output = "Right"
    assert str(actual_output) == expected_output


def test_problem_p03477_3():
    actual_output = problem_p03477("3 4 5 2")
    expected_output = "Balanced"
    assert str(actual_output) == expected_output
