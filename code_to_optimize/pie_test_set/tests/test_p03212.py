from code_to_optimize.pie_test_set.p03212 import problem_p03212


def test_problem_p03212_0():
    actual_output = problem_p03212("575")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03212_1():
    actual_output = problem_p03212("575")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03212_2():
    actual_output = problem_p03212("3600")
    expected_output = "13"
    assert str(actual_output) == expected_output


def test_problem_p03212_3():
    actual_output = problem_p03212("999999999")
    expected_output = "26484"
    assert str(actual_output) == expected_output
