from code_to_optimize.pie_test_set.p03679 import problem_p03679


def test_problem_p03679_0():
    actual_output = problem_p03679("4 3 6")
    expected_output = "safe"
    assert str(actual_output) == expected_output


def test_problem_p03679_1():
    actual_output = problem_p03679("4 3 6")
    expected_output = "safe"
    assert str(actual_output) == expected_output


def test_problem_p03679_2():
    actual_output = problem_p03679("6 5 1")
    expected_output = "delicious"
    assert str(actual_output) == expected_output


def test_problem_p03679_3():
    actual_output = problem_p03679("3 7 12")
    expected_output = "dangerous"
    assert str(actual_output) == expected_output
