from pie_test_set.p02970 import problem_p02970


def test_problem_p02970_0():
    actual_output = problem_p02970("6 2")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p02970_1():
    actual_output = problem_p02970("20 4")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02970_2():
    actual_output = problem_p02970("14 3")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p02970_3():
    actual_output = problem_p02970("6 2")
    expected_output = "2"
    assert str(actual_output) == expected_output
