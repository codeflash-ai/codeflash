from pie_test_set.p03480 import problem_p03480


def test_problem_p03480_0():
    actual_output = problem_p03480("010")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03480_1():
    actual_output = problem_p03480("100000000")
    expected_output = "8"
    assert str(actual_output) == expected_output


def test_problem_p03480_2():
    actual_output = problem_p03480("010")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03480_3():
    actual_output = problem_p03480("00001111")
    expected_output = "4"
    assert str(actual_output) == expected_output
