from pie_test_set.p03834 import problem_p03834


def test_problem_p03834_0():
    actual_output = problem_p03834("happy,newyear,enjoy")
    expected_output = "happy newyear enjoy"
    assert str(actual_output) == expected_output


def test_problem_p03834_1():
    actual_output = problem_p03834("haiku,atcoder,tasks")
    expected_output = "haiku atcoder tasks"
    assert str(actual_output) == expected_output


def test_problem_p03834_2():
    actual_output = problem_p03834("abcde,fghihgf,edcba")
    expected_output = "abcde fghihgf edcba"
    assert str(actual_output) == expected_output


def test_problem_p03834_3():
    actual_output = problem_p03834("happy,newyear,enjoy")
    expected_output = "happy newyear enjoy"
    assert str(actual_output) == expected_output
