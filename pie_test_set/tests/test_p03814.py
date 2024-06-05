from pie_test_set.p03814 import problem_p03814


def test_problem_p03814_0():
    actual_output = problem_p03814("QWERTYASDFZXCV")
    expected_output = "5"
    assert str(actual_output) == expected_output


def test_problem_p03814_1():
    actual_output = problem_p03814("HASFJGHOGAKZZFEGA")
    expected_output = "12"
    assert str(actual_output) == expected_output


def test_problem_p03814_2():
    actual_output = problem_p03814("ZABCZ")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03814_3():
    actual_output = problem_p03814("QWERTYASDFZXCV")
    expected_output = "5"
    assert str(actual_output) == expected_output
