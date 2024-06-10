from pie_test_set.p03631 import problem_p03631


def test_problem_p03631_0():
    actual_output = problem_p03631("575")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03631_1():
    actual_output = problem_p03631("812")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p03631_2():
    actual_output = problem_p03631("575")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03631_3():
    actual_output = problem_p03631("123")
    expected_output = "No"
    assert str(actual_output) == expected_output
