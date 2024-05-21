from pie_test_set.p03359 import problem_p03359


def test_problem_p03359_0():
    actual_output = problem_p03359("5 5")
    expected_output = "5"
    assert str(actual_output) == expected_output


def test_problem_p03359_1():
    actual_output = problem_p03359("11 30")
    expected_output = "11"
    assert str(actual_output) == expected_output


def test_problem_p03359_2():
    actual_output = problem_p03359("2 1")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03359_3():
    actual_output = problem_p03359("5 5")
    expected_output = "5"
    assert str(actual_output) == expected_output
