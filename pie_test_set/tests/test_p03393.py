from pie_test_set.p03393 import problem_p03393


def test_problem_p03393_0():
    actual_output = problem_p03393("atcoder")
    expected_output = "atcoderb"
    assert str(actual_output) == expected_output


def test_problem_p03393_1():
    actual_output = problem_p03393("zyxwvutsrqponmlkjihgfedcba")
    expected_output = "-1"
    assert str(actual_output) == expected_output


def test_problem_p03393_2():
    actual_output = problem_p03393("atcoder")
    expected_output = "atcoderb"
    assert str(actual_output) == expected_output


def test_problem_p03393_3():
    actual_output = problem_p03393("abcdefghijklmnopqrstuvwzyx")
    expected_output = "abcdefghijklmnopqrstuvx"
    assert str(actual_output) == expected_output


def test_problem_p03393_4():
    actual_output = problem_p03393("abc")
    expected_output = "abcd"
    assert str(actual_output) == expected_output
