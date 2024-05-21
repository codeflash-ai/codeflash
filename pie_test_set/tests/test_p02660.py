from pie_test_set.p02660 import problem_p02660


def test_problem_p02660_0():
    actual_output = problem_p02660("24")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02660_1():
    actual_output = problem_p02660("997764507000")
    expected_output = "7"
    assert str(actual_output) == expected_output


def test_problem_p02660_2():
    actual_output = problem_p02660("1")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p02660_3():
    actual_output = problem_p02660("1000000007")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p02660_4():
    actual_output = problem_p02660("64")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02660_5():
    actual_output = problem_p02660("24")
    expected_output = "3"
    assert str(actual_output) == expected_output
