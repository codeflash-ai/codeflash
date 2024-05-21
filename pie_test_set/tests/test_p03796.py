from pie_test_set.p03796 import problem_p03796


def test_problem_p03796_0():
    actual_output = problem_p03796("3")
    expected_output = "6"
    assert str(actual_output) == expected_output


def test_problem_p03796_1():
    actual_output = problem_p03796("100000")
    expected_output = "457992974"
    assert str(actual_output) == expected_output


def test_problem_p03796_2():
    actual_output = problem_p03796("3")
    expected_output = "6"
    assert str(actual_output) == expected_output


def test_problem_p03796_3():
    actual_output = problem_p03796("10")
    expected_output = "3628800"
    assert str(actual_output) == expected_output
