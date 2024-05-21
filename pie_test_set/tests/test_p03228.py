from pie_test_set.p03228 import problem_p03228


def test_problem_p03228_0():
    actual_output = problem_p03228("5 4 2")
    expected_output = "5 3"
    assert str(actual_output) == expected_output


def test_problem_p03228_1():
    actual_output = problem_p03228("3 3 3")
    expected_output = "1 3"
    assert str(actual_output) == expected_output


def test_problem_p03228_2():
    actual_output = problem_p03228("5 4 2")
    expected_output = "5 3"
    assert str(actual_output) == expected_output


def test_problem_p03228_3():
    actual_output = problem_p03228("314159265 358979323 84")
    expected_output = "448759046 224379523"
    assert str(actual_output) == expected_output
