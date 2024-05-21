from pie_test_set.p03697 import problem_p03697


def test_problem_p03697_0():
    actual_output = problem_p03697("6 3")
    expected_output = "9"
    assert str(actual_output) == expected_output


def test_problem_p03697_1():
    actual_output = problem_p03697("6 4")
    expected_output = "error"
    assert str(actual_output) == expected_output


def test_problem_p03697_2():
    actual_output = problem_p03697("6 3")
    expected_output = "9"
    assert str(actual_output) == expected_output
