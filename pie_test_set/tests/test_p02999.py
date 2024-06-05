from pie_test_set.p02999 import problem_p02999


def test_problem_p02999_0():
    actual_output = problem_p02999("3 5")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p02999_1():
    actual_output = problem_p02999("6 6")
    expected_output = "10"
    assert str(actual_output) == expected_output


def test_problem_p02999_2():
    actual_output = problem_p02999("7 5")
    expected_output = "10"
    assert str(actual_output) == expected_output


def test_problem_p02999_3():
    actual_output = problem_p02999("3 5")
    expected_output = "0"
    assert str(actual_output) == expected_output
