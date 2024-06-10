from pie_test_set.p02842 import problem_p02842


def test_problem_p02842_0():
    actual_output = problem_p02842("432")
    expected_output = "400"
    assert str(actual_output) == expected_output


def test_problem_p02842_1():
    actual_output = problem_p02842("1001")
    expected_output = "927"
    assert str(actual_output) == expected_output


def test_problem_p02842_2():
    actual_output = problem_p02842("1079")
    expected_output = ":("
    assert str(actual_output) == expected_output


def test_problem_p02842_3():
    actual_output = problem_p02842("432")
    expected_output = "400"
    assert str(actual_output) == expected_output
