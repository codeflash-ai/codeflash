from pie_test_set.p02577 import problem_p02577


def test_problem_p02577_0():
    actual_output = problem_p02577("123456789")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02577_1():
    actual_output = problem_p02577("123456789")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02577_2():
    actual_output = problem_p02577("0")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02577_3():
    actual_output = problem_p02577(
        "31415926535897932384626433832795028841971693993751058209749445923078164062862089986280"
    )
    expected_output = "No"
    assert str(actual_output) == expected_output
