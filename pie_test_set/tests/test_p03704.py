from pie_test_set.p03704 import problem_p03704


def test_problem_p03704_0():
    actual_output = problem_p03704("63")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03704_1():
    actual_output = problem_p03704("63")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03704_2():
    actual_output = problem_p03704("864197532")
    expected_output = "1920"
    assert str(actual_output) == expected_output


def test_problem_p03704_3():
    actual_output = problem_p03704("75")
    expected_output = "0"
    assert str(actual_output) == expected_output
