from pie_test_set.p02570 import problem_p02570


def test_problem_p02570_0():
    actual_output = problem_p02570("1000 15 80")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02570_1():
    actual_output = problem_p02570("2000 20 100")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02570_2():
    actual_output = problem_p02570("10000 1 1")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p02570_3():
    actual_output = problem_p02570("1000 15 80")
    expected_output = "Yes"
    assert str(actual_output) == expected_output
