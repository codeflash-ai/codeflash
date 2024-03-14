from code_to_optimize.pie_test_set.p03211 import problem_p03211


def test_problem_p03211_0():
    actual_output = problem_p03211("1234567876")
    expected_output = "34"
    assert str(actual_output) == expected_output


def test_problem_p03211_1():
    actual_output = problem_p03211("35753")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03211_2():
    actual_output = problem_p03211("1234567876")
    expected_output = "34"
    assert str(actual_output) == expected_output


def test_problem_p03211_3():
    actual_output = problem_p03211("1111111111")
    expected_output = "642"
    assert str(actual_output) == expected_output
