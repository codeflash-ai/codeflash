from code_to_optimize.pie_test_set.p03803 import problem_p03803


def test_problem_p03803_0():
    actual_output = problem_p03803("8 6")
    expected_output = "Alice"
    assert str(actual_output) == expected_output


def test_problem_p03803_1():
    actual_output = problem_p03803("8 6")
    expected_output = "Alice"
    assert str(actual_output) == expected_output


def test_problem_p03803_2():
    actual_output = problem_p03803("1 1")
    expected_output = "Draw"
    assert str(actual_output) == expected_output


def test_problem_p03803_3():
    actual_output = problem_p03803("13 1")
    expected_output = "Bob"
    assert str(actual_output) == expected_output
