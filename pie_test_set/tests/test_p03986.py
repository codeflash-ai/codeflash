from pie_test_set.p03986 import problem_p03986


def test_problem_p03986_0():
    actual_output = problem_p03986("TSTTSS")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03986_1():
    actual_output = problem_p03986("TSTTSS")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03986_2():
    actual_output = problem_p03986("SSTTST")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03986_3():
    actual_output = problem_p03986("TSSTTTSS")
    expected_output = "4"
    assert str(actual_output) == expected_output
