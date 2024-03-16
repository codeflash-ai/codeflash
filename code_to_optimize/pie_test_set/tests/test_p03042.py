from code_to_optimize.pie_test_set.p03042 import problem_p03042


def test_problem_p03042_0():
    actual_output = problem_p03042("1905")
    expected_output = "YYMM"
    assert str(actual_output) == expected_output


def test_problem_p03042_1():
    actual_output = problem_p03042("1905")
    expected_output = "YYMM"
    assert str(actual_output) == expected_output


def test_problem_p03042_2():
    actual_output = problem_p03042("1700")
    expected_output = "NA"
    assert str(actual_output) == expected_output


def test_problem_p03042_3():
    actual_output = problem_p03042("0112")
    expected_output = "AMBIGUOUS"
    assert str(actual_output) == expected_output
