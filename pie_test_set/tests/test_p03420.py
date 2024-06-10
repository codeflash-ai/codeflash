from pie_test_set.p03420 import problem_p03420


def test_problem_p03420_0():
    actual_output = problem_p03420("5 2")
    expected_output = "7"
    assert str(actual_output) == expected_output


def test_problem_p03420_1():
    actual_output = problem_p03420("5 2")
    expected_output = "7"
    assert str(actual_output) == expected_output


def test_problem_p03420_2():
    actual_output = problem_p03420("31415 9265")
    expected_output = "287927211"
    assert str(actual_output) == expected_output


def test_problem_p03420_3():
    actual_output = problem_p03420("10 0")
    expected_output = "100"
    assert str(actual_output) == expected_output
