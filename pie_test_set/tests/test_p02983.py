from pie_test_set.p02983 import problem_p02983


def test_problem_p02983_0():
    actual_output = problem_p02983("2020 2040")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p02983_1():
    actual_output = problem_p02983("4 5")
    expected_output = "20"
    assert str(actual_output) == expected_output


def test_problem_p02983_2():
    actual_output = problem_p02983("2020 2040")
    expected_output = "2"
    assert str(actual_output) == expected_output
