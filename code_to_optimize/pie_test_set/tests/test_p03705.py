from code_to_optimize.pie_test_set.p03705 import problem_p03705


def test_problem_p03705_0():
    actual_output = problem_p03705("4 4 6")
    expected_output = "5"
    assert str(actual_output) == expected_output


def test_problem_p03705_1():
    actual_output = problem_p03705("5 4 3")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03705_2():
    actual_output = problem_p03705("1 7 10")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03705_3():
    actual_output = problem_p03705("1 3 3")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03705_4():
    actual_output = problem_p03705("4 4 6")
    expected_output = "5"
    assert str(actual_output) == expected_output
