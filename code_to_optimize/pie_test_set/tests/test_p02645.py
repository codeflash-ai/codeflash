from code_to_optimize.pie_test_set.p02645 import problem_p02645


def test_problem_p02645_0():
    actual_output = problem_p02645("takahashi")
    expected_output = "tak"
    assert str(actual_output) == expected_output


def test_problem_p02645_1():
    actual_output = problem_p02645("naohiro")
    expected_output = "nao"
    assert str(actual_output) == expected_output


def test_problem_p02645_2():
    actual_output = problem_p02645("takahashi")
    expected_output = "tak"
    assert str(actual_output) == expected_output
