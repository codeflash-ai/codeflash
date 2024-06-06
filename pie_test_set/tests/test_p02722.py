from pie_test_set.p02722 import problem_p02722


def test_problem_p02722_0():
    actual_output = problem_p02722("6")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02722_1():
    actual_output = problem_p02722("314159265358")
    expected_output = "9"
    assert str(actual_output) == expected_output


def test_problem_p02722_2():
    actual_output = problem_p02722("6")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02722_3():
    actual_output = problem_p02722("3141")
    expected_output = "13"
    assert str(actual_output) == expected_output
