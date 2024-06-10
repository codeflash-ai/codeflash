from pie_test_set.p04001 import problem_p04001


def test_problem_p04001_0():
    actual_output = problem_p04001("125")
    expected_output = "176"
    assert str(actual_output) == expected_output
