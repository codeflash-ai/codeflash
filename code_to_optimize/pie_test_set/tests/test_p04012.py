from code_to_optimize.pie_test_set.p04012 import problem_p04012


def test_problem_p04012_0():
    actual_output = problem_p04012("abaccaba")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p04012_1():
    actual_output = problem_p04012("abaccaba")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p04012_2():
    actual_output = problem_p04012("hthth")
    expected_output = "No"
    assert str(actual_output) == expected_output
