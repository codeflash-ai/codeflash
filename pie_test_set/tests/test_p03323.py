from pie_test_set.p03323 import problem_p03323


def test_problem_p03323_0():
    actual_output = problem_p03323("5 4")
    expected_output = "Yay!"
    assert str(actual_output) == expected_output


def test_problem_p03323_1():
    actual_output = problem_p03323("11 4")
    expected_output = ":("
    assert str(actual_output) == expected_output


def test_problem_p03323_2():
    actual_output = problem_p03323("8 8")
    expected_output = "Yay!"
    assert str(actual_output) == expected_output


def test_problem_p03323_3():
    actual_output = problem_p03323("5 4")
    expected_output = "Yay!"
    assert str(actual_output) == expected_output
