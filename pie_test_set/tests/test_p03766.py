from pie_test_set.p03766 import problem_p03766


def test_problem_p03766_0():
    actual_output = problem_p03766("2")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03766_1():
    actual_output = problem_p03766("654321")
    expected_output = "968545283"
    assert str(actual_output) == expected_output


def test_problem_p03766_2():
    actual_output = problem_p03766("2")
    expected_output = "4"
    assert str(actual_output) == expected_output
