from code_to_optimize.pie_test_set.p03943 import problem_p03943


def test_problem_p03943_0():
    actual_output = problem_p03943("10 30 20")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03943_1():
    actual_output = problem_p03943("56 25 31")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03943_2():
    actual_output = problem_p03943("10 30 20")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03943_3():
    actual_output = problem_p03943("30 30 100")
    expected_output = "No"
    assert str(actual_output) == expected_output
