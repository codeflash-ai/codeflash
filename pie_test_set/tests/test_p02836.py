from pie_test_set.p02836 import problem_p02836


def test_problem_p02836_0():
    actual_output = problem_p02836("redcoder")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p02836_1():
    actual_output = problem_p02836("abcdabc")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p02836_2():
    actual_output = problem_p02836("redcoder")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p02836_3():
    actual_output = problem_p02836("vvvvvv")
    expected_output = "0"
    assert str(actual_output) == expected_output
