from pie_test_set.p04040 import problem_p04040


def test_problem_p04040_0():
    actual_output = problem_p04040("2 3 1 1")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p04040_1():
    actual_output = problem_p04040("2 3 1 1")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p04040_2():
    actual_output = problem_p04040("100000 100000 99999 99999")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p04040_3():
    actual_output = problem_p04040("100000 100000 44444 55555")
    expected_output = "738162020"
    assert str(actual_output) == expected_output


def test_problem_p04040_4():
    actual_output = problem_p04040("10 7 3 4")
    expected_output = "3570"
    assert str(actual_output) == expected_output
