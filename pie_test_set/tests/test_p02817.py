from pie_test_set.p02817 import problem_p02817


def test_problem_p02817_0():
    actual_output = problem_p02817("oder atc")
    expected_output = "atcoder"
    assert str(actual_output) == expected_output


def test_problem_p02817_1():
    actual_output = problem_p02817("humu humu")
    expected_output = "humuhumu"
    assert str(actual_output) == expected_output


def test_problem_p02817_2():
    actual_output = problem_p02817("oder atc")
    expected_output = "atcoder"
    assert str(actual_output) == expected_output
