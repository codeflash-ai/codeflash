from code_to_optimize.pie_test_set.p03407 import problem_p03407


def test_problem_p03407_0():
    actual_output = problem_p03407("50 100 120")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03407_1():
    actual_output = problem_p03407("500 100 1000")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p03407_2():
    actual_output = problem_p03407("19 123 143")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p03407_3():
    actual_output = problem_p03407("19 123 142")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03407_4():
    actual_output = problem_p03407("50 100 120")
    expected_output = "Yes"
    assert str(actual_output) == expected_output
