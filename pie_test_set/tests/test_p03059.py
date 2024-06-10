from pie_test_set.p03059 import problem_p03059


def test_problem_p03059_0():
    actual_output = problem_p03059("3 5 7")
    expected_output = "10"
    assert str(actual_output) == expected_output


def test_problem_p03059_1():
    actual_output = problem_p03059("3 5 7")
    expected_output = "10"
    assert str(actual_output) == expected_output


def test_problem_p03059_2():
    actual_output = problem_p03059("20 20 19")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03059_3():
    actual_output = problem_p03059("3 2 9")
    expected_output = "6"
    assert str(actual_output) == expected_output
