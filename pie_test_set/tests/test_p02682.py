from pie_test_set.p02682 import problem_p02682


def test_problem_p02682_0():
    actual_output = problem_p02682("2 1 1 3")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p02682_1():
    actual_output = problem_p02682("2000000000 0 0 2000000000")
    expected_output = "2000000000"
    assert str(actual_output) == expected_output


def test_problem_p02682_2():
    actual_output = problem_p02682("1 2 3 4")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p02682_3():
    actual_output = problem_p02682("2 1 1 3")
    expected_output = "2"
    assert str(actual_output) == expected_output
