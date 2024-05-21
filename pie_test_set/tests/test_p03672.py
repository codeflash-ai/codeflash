from pie_test_set.p03672 import problem_p03672


def test_problem_p03672_0():
    actual_output = problem_p03672("abaababaab")
    expected_output = "6"
    assert str(actual_output) == expected_output


def test_problem_p03672_1():
    actual_output = problem_p03672("abaababaab")
    expected_output = "6"
    assert str(actual_output) == expected_output


def test_problem_p03672_2():
    actual_output = problem_p03672("abcabcabcabc")
    expected_output = "6"
    assert str(actual_output) == expected_output


def test_problem_p03672_3():
    actual_output = problem_p03672("xxxx")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03672_4():
    actual_output = problem_p03672("akasakaakasakasakaakas")
    expected_output = "14"
    assert str(actual_output) == expected_output
