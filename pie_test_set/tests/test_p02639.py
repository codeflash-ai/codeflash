from pie_test_set.p02639 import problem_p02639


def test_problem_p02639_0():
    actual_output = problem_p02639("0 2 3 4 5")
    expected_output = "1"
    assert str(actual_output) == expected_output
