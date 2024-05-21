from pie_test_set.p04033 import problem_p04033


def test_problem_p04033_0():
    actual_output = problem_p04033("1 3")
    expected_output = "Positive"
    assert str(actual_output) == expected_output
