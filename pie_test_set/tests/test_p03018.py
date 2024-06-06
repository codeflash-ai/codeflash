from pie_test_set.p03018 import problem_p03018


def test_problem_p03018_0():
    actual_output = problem_p03018("ABCABC")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03018_1():
    actual_output = problem_p03018("ABCACCBABCBCAABCB")
    expected_output = "6"
    assert str(actual_output) == expected_output


def test_problem_p03018_2():
    actual_output = problem_p03018("C")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03018_3():
    actual_output = problem_p03018("ABCABC")
    expected_output = "3"
    assert str(actual_output) == expected_output
