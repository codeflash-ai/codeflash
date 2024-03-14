from code_to_optimize.pie_test_set.p03071 import problem_p03071


def test_problem_p03071_0():
    actual_output = problem_p03071("5 3")
    expected_output = "9"
    assert str(actual_output) == expected_output


def test_problem_p03071_1():
    actual_output = problem_p03071("3 4")
    expected_output = "7"
    assert str(actual_output) == expected_output


def test_problem_p03071_2():
    actual_output = problem_p03071("5 3")
    expected_output = "9"
    assert str(actual_output) == expected_output


def test_problem_p03071_3():
    actual_output = problem_p03071("6 6")
    expected_output = "12"
    assert str(actual_output) == expected_output
