from pie_test_set.p03196 import problem_p03196


def test_problem_p03196_0():
    actual_output = problem_p03196("3 24")
    expected_output = "2"
    assert str(actual_output) == expected_output
