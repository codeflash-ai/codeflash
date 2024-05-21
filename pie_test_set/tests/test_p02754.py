from pie_test_set.p02754 import problem_p02754


def test_problem_p02754_0():
    actual_output = problem_p02754("8 3 4")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p02754_1():
    actual_output = problem_p02754("8 0 4")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p02754_2():
    actual_output = problem_p02754("8 3 4")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p02754_3():
    actual_output = problem_p02754("6 2 4")
    expected_output = "2"
    assert str(actual_output) == expected_output
