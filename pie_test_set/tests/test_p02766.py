from pie_test_set.p02766 import problem_p02766


def test_problem_p02766_0():
    actual_output = problem_p02766("11 2")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p02766_1():
    actual_output = problem_p02766("11 2")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p02766_2():
    actual_output = problem_p02766("314159265 3")
    expected_output = "18"
    assert str(actual_output) == expected_output


def test_problem_p02766_3():
    actual_output = problem_p02766("1010101 10")
    expected_output = "7"
    assert str(actual_output) == expected_output
