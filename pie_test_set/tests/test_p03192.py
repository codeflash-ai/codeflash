from pie_test_set.p03192 import problem_p03192


def test_problem_p03192_0():
    actual_output = problem_p03192("1222")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03192_1():
    actual_output = problem_p03192("9592")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03192_2():
    actual_output = problem_p03192("3456")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03192_3():
    actual_output = problem_p03192("1222")
    expected_output = "3"
    assert str(actual_output) == expected_output
