from pie_test_set.p02635 import problem_p02635


def test_problem_p02635_0():
    actual_output = problem_p02635("0101 1")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p02635_1():
    actual_output = problem_p02635("0101 1")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p02635_2():
    actual_output = problem_p02635("01100110 2")
    expected_output = "14"
    assert str(actual_output) == expected_output


def test_problem_p02635_3():
    actual_output = problem_p02635(
        "1101010010101101110111100011011111011000111101110101010010101010101 20"
    )
    expected_output = "113434815"
    assert str(actual_output) == expected_output
