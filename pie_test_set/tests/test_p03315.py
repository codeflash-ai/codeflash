from pie_test_set.p03315 import problem_p03315


def test_problem_p03315_0():
    actual_output = problem_p03315("+-++")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03315_1():
    actual_output = problem_p03315("+-++")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03315_2():
    actual_output = problem_p03315("-+--")
    expected_output = "-2"
    assert str(actual_output) == expected_output


def test_problem_p03315_3():
    actual_output = problem_p03315("----")
    expected_output = "-4"
    assert str(actual_output) == expected_output
