from pie_test_set.p03456 import problem_p03456


def test_problem_p03456_0():
    actual_output = problem_p03456("1 21")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03456_1():
    actual_output = problem_p03456("1 21")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03456_2():
    actual_output = problem_p03456("12 10")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p03456_3():
    actual_output = problem_p03456("100 100")
    expected_output = "No"
    assert str(actual_output) == expected_output
