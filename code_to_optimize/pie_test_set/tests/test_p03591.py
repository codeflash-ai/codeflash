from code_to_optimize.pie_test_set.p03591 import problem_p03591


def test_problem_p03591_0():
    actual_output = problem_p03591("YAKINIKU")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03591_1():
    actual_output = problem_p03591("YAKINIKU")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03591_2():
    actual_output = problem_p03591("TAKOYAKI")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p03591_3():
    actual_output = problem_p03591("YAK")
    expected_output = "No"
    assert str(actual_output) == expected_output
