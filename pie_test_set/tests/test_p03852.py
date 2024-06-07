from pie_test_set.p03852 import problem_p03852


def test_problem_p03852_0():
    actual_output = problem_p03852("a")
    expected_output = "vowel"
    assert str(actual_output) == expected_output


def test_problem_p03852_1():
    actual_output = problem_p03852("s")
    expected_output = "consonant"
    assert str(actual_output) == expected_output


def test_problem_p03852_2():
    actual_output = problem_p03852("z")
    expected_output = "consonant"
    assert str(actual_output) == expected_output


def test_problem_p03852_3():
    actual_output = problem_p03852("a")
    expected_output = "vowel"
    assert str(actual_output) == expected_output
