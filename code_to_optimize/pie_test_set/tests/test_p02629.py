from code_to_optimize.pie_test_set.p02629 import problem_p02629


def test_problem_p02629_0():
    actual_output = problem_p02629("2")
    expected_output = "b"
    assert str(actual_output) == expected_output


def test_problem_p02629_1():
    actual_output = problem_p02629("27")
    expected_output = "aa"
    assert str(actual_output) == expected_output


def test_problem_p02629_2():
    actual_output = problem_p02629("2")
    expected_output = "b"
    assert str(actual_output) == expected_output


def test_problem_p02629_3():
    actual_output = problem_p02629("123456789")
    expected_output = "jjddja"
    assert str(actual_output) == expected_output
