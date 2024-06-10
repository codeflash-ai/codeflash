from pie_test_set.p03250 import problem_p03250


def test_problem_p03250_0():
    actual_output = problem_p03250("1 5 2")
    expected_output = "53"
    assert str(actual_output) == expected_output


def test_problem_p03250_1():
    actual_output = problem_p03250("9 9 9")
    expected_output = "108"
    assert str(actual_output) == expected_output


def test_problem_p03250_2():
    actual_output = problem_p03250("1 5 2")
    expected_output = "53"
    assert str(actual_output) == expected_output


def test_problem_p03250_3():
    actual_output = problem_p03250("6 6 7")
    expected_output = "82"
    assert str(actual_output) == expected_output
