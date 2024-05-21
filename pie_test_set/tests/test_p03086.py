from pie_test_set.p03086 import problem_p03086


def test_problem_p03086_0():
    actual_output = problem_p03086("ATCODER")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03086_1():
    actual_output = problem_p03086("SHINJUKU")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03086_2():
    actual_output = problem_p03086("ATCODER")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03086_3():
    actual_output = problem_p03086("HATAGAYA")
    expected_output = "5"
    assert str(actual_output) == expected_output
