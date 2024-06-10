from pie_test_set.p03016 import problem_p03016


def test_problem_p03016_0():
    actual_output = problem_p03016("5 3 4 10007")
    expected_output = "5563"
    assert str(actual_output) == expected_output


def test_problem_p03016_1():
    actual_output = problem_p03016("4 8 1 1000000")
    expected_output = "891011"
    assert str(actual_output) == expected_output


def test_problem_p03016_2():
    actual_output = problem_p03016("107 10000000000007 1000000000000007 998244353")
    expected_output = "39122908"
    assert str(actual_output) == expected_output


def test_problem_p03016_3():
    actual_output = problem_p03016("5 3 4 10007")
    expected_output = "5563"
    assert str(actual_output) == expected_output
