from pie_test_set.p02584 import problem_p02584


def test_problem_p02584_0():
    actual_output = problem_p02584("6 2 4")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p02584_1():
    actual_output = problem_p02584("1000000000000000 1000000000000000 1000000000000000")
    expected_output = "1000000000000000"
    assert str(actual_output) == expected_output


def test_problem_p02584_2():
    actual_output = problem_p02584("6 2 4")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p02584_3():
    actual_output = problem_p02584("10 1 2")
    expected_output = "8"
    assert str(actual_output) == expected_output


def test_problem_p02584_4():
    actual_output = problem_p02584("7 4 3")
    expected_output = "1"
    assert str(actual_output) == expected_output
