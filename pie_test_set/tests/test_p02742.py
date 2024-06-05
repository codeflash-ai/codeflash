from pie_test_set.p02742 import problem_p02742


def test_problem_p02742_0():
    actual_output = problem_p02742("4 5")
    expected_output = "10"
    assert str(actual_output) == expected_output


def test_problem_p02742_1():
    actual_output = problem_p02742("7 3")
    expected_output = "11"
    assert str(actual_output) == expected_output


def test_problem_p02742_2():
    actual_output = problem_p02742("1000000000 1000000000")
    expected_output = "500000000000000000"
    assert str(actual_output) == expected_output


def test_problem_p02742_3():
    actual_output = problem_p02742("4 5")
    expected_output = "10"
    assert str(actual_output) == expected_output
