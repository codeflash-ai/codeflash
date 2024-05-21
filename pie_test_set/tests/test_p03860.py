from pie_test_set.p03860 import problem_p03860


def test_problem_p03860_0():
    actual_output = problem_p03860("AtCoder Beginner Contest")
    expected_output = "ABC"
    assert str(actual_output) == expected_output


def test_problem_p03860_1():
    actual_output = problem_p03860("AtCoder Beginner Contest")
    expected_output = "ABC"
    assert str(actual_output) == expected_output


def test_problem_p03860_2():
    actual_output = problem_p03860("AtCoder Snuke Contest")
    expected_output = "ASC"
    assert str(actual_output) == expected_output


def test_problem_p03860_3():
    actual_output = problem_p03860("AtCoder X Contest")
    expected_output = "AXC"
    assert str(actual_output) == expected_output
