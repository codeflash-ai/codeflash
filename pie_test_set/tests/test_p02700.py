from pie_test_set.p02700 import problem_p02700


def test_problem_p02700_0():
    actual_output = problem_p02700("10 9 10 10")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p02700_1():
    actual_output = problem_p02700("46 4 40 5")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02700_2():
    actual_output = problem_p02700("10 9 10 10")
    expected_output = "No"
    assert str(actual_output) == expected_output
