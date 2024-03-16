from code_to_optimize.pie_test_set.p04019 import problem_p04019


def test_problem_p04019_0():
    actual_output = problem_p04019("SENW")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p04019_1():
    actual_output = problem_p04019("NSNNSNSN")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p04019_2():
    actual_output = problem_p04019("NNEW")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p04019_3():
    actual_output = problem_p04019("SENW")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p04019_4():
    actual_output = problem_p04019("W")
    expected_output = "No"
    assert str(actual_output) == expected_output
