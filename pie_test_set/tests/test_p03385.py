from pie_test_set.p03385 import problem_p03385


def test_problem_p03385_0():
    actual_output = problem_p03385("bac")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03385_1():
    actual_output = problem_p03385("bab")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p03385_2():
    actual_output = problem_p03385("abc")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03385_3():
    actual_output = problem_p03385("aaa")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p03385_4():
    actual_output = problem_p03385("bac")
    expected_output = "Yes"
    assert str(actual_output) == expected_output
