from pie_test_set.p03373 import problem_p03373


def test_problem_p03373_0():
    actual_output = problem_p03373("1500 2000 1600 3 2")
    expected_output = "7900"
    assert str(actual_output) == expected_output
