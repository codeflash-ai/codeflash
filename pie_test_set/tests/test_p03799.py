from pie_test_set.p03799 import problem_p03799


def test_problem_p03799_0():
    actual_output = problem_p03799("1 6")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03799_1():
    actual_output = problem_p03799("12345 678901")
    expected_output = "175897"
    assert str(actual_output) == expected_output


def test_problem_p03799_2():
    actual_output = problem_p03799("1 6")
    expected_output = "2"
    assert str(actual_output) == expected_output
