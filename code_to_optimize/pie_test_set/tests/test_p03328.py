from code_to_optimize.pie_test_set.p03328 import problem_p03328


def test_problem_p03328_0():
    actual_output = problem_p03328("8 13")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03328_1():
    actual_output = problem_p03328("54 65")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03328_2():
    actual_output = problem_p03328("8 13")
    expected_output = "2"
    assert str(actual_output) == expected_output
