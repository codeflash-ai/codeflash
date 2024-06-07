from pie_test_set.p03304 import problem_p03304


def test_problem_p03304_0():
    actual_output = problem_p03304("2 3 1")
    expected_output = "1.0000000000"
    assert str(actual_output) == expected_output


def test_problem_p03304_1():
    actual_output = problem_p03304("2 3 1")
    expected_output = "1.0000000000"
    assert str(actual_output) == expected_output


def test_problem_p03304_2():
    actual_output = problem_p03304("1000000000 180707 0")
    expected_output = "0.0001807060"
    assert str(actual_output) == expected_output
