from code_to_optimize.pie_test_set.p03463 import problem_p03463


def test_problem_p03463_0():
    actual_output = problem_p03463("5 2 4")
    expected_output = "Alice"
    assert str(actual_output) == expected_output


def test_problem_p03463_1():
    actual_output = problem_p03463("58 23 42")
    expected_output = "Borys"
    assert str(actual_output) == expected_output


def test_problem_p03463_2():
    actual_output = problem_p03463("2 1 2")
    expected_output = "Borys"
    assert str(actual_output) == expected_output


def test_problem_p03463_3():
    actual_output = problem_p03463("5 2 4")
    expected_output = "Alice"
    assert str(actual_output) == expected_output
