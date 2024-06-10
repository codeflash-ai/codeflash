from pie_test_set.p04005 import problem_p04005


def test_problem_p04005_0():
    actual_output = problem_p04005("3 3 3")
    expected_output = "9"
    assert str(actual_output) == expected_output


def test_problem_p04005_1():
    actual_output = problem_p04005("2 2 4")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p04005_2():
    actual_output = problem_p04005("3 3 3")
    expected_output = "9"
    assert str(actual_output) == expected_output


def test_problem_p04005_3():
    actual_output = problem_p04005("5 3 5")
    expected_output = "15"
    assert str(actual_output) == expected_output
