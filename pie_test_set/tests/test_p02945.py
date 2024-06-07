from pie_test_set.p02945 import problem_p02945


def test_problem_p02945_0():
    actual_output = problem_p02945("-13 3")
    expected_output = "-10"
    assert str(actual_output) == expected_output
