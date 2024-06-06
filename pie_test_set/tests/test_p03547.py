from pie_test_set.p03547 import problem_p03547


def test_problem_p03547_0():
    actual_output = problem_p03547("A B")
    expected_output = "<"
    assert str(actual_output) == expected_output


def test_problem_p03547_1():
    actual_output = problem_p03547("F F")
    expected_output = "="
    assert str(actual_output) == expected_output


def test_problem_p03547_2():
    actual_output = problem_p03547("A B")
    expected_output = "<"
    assert str(actual_output) == expected_output


def test_problem_p03547_3():
    actual_output = problem_p03547("E C")
    expected_output = ">"
    assert str(actual_output) == expected_output
