from pie_test_set.p03601 import problem_p03601


def test_problem_p03601_0():
    actual_output = problem_p03601("1 2 10 20 15 200")
    expected_output = "110 10"
    assert str(actual_output) == expected_output
