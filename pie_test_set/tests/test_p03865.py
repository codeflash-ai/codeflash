from pie_test_set.p03865 import problem_p03865


def test_problem_p03865_0():
    actual_output = problem_p03865("aba")
    expected_output = "Second"
    assert str(actual_output) == expected_output
