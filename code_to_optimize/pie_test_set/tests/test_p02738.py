from code_to_optimize.pie_test_set.p02738 import problem_p02738


def test_problem_p02738_0():
    actual_output = problem_p02738("1 998244353")
    expected_output = "6"
    assert str(actual_output) == expected_output


def test_problem_p02738_1():
    actual_output = problem_p02738("314 1000000007")
    expected_output = "182908545"
    assert str(actual_output) == expected_output


def test_problem_p02738_2():
    actual_output = problem_p02738("1 998244353")
    expected_output = "6"
    assert str(actual_output) == expected_output


def test_problem_p02738_3():
    actual_output = problem_p02738("2 998244353")
    expected_output = "261"
    assert str(actual_output) == expected_output
