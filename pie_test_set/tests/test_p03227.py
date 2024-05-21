from pie_test_set.p03227 import problem_p03227


def test_problem_p03227_0():
    actual_output = problem_p03227("abc")
    expected_output = "cba"
    assert str(actual_output) == expected_output


def test_problem_p03227_1():
    actual_output = problem_p03227("abc")
    expected_output = "cba"
    assert str(actual_output) == expected_output


def test_problem_p03227_2():
    actual_output = problem_p03227("ac")
    expected_output = "ac"
    assert str(actual_output) == expected_output
