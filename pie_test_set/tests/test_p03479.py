from pie_test_set.p03479 import problem_p03479


def test_problem_p03479_0():
    actual_output = problem_p03479("3 20")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03479_1():
    actual_output = problem_p03479("3 20")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03479_2():
    actual_output = problem_p03479("314159265 358979323846264338")
    expected_output = "31"
    assert str(actual_output) == expected_output


def test_problem_p03479_3():
    actual_output = problem_p03479("25 100")
    expected_output = "3"
    assert str(actual_output) == expected_output
