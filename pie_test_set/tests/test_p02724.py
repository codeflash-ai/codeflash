from pie_test_set.p02724 import problem_p02724


def test_problem_p02724_0():
    actual_output = problem_p02724("1024")
    expected_output = "2020"
    assert str(actual_output) == expected_output


def test_problem_p02724_1():
    actual_output = problem_p02724("0")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p02724_2():
    actual_output = problem_p02724("1000000000")
    expected_output = "2000000000"
    assert str(actual_output) == expected_output


def test_problem_p02724_3():
    actual_output = problem_p02724("1024")
    expected_output = "2020"
    assert str(actual_output) == expected_output
