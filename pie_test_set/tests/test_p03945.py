from pie_test_set.p03945 import problem_p03945


def test_problem_p03945_0():
    actual_output = problem_p03945("BBBWW")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03945_1():
    actual_output = problem_p03945("WWWWWW")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03945_2():
    actual_output = problem_p03945("WBWBWBWBWB")
    expected_output = "9"
    assert str(actual_output) == expected_output


def test_problem_p03945_3():
    actual_output = problem_p03945("BBBWW")
    expected_output = "1"
    assert str(actual_output) == expected_output
