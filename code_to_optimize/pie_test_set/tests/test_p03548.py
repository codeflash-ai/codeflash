from code_to_optimize.pie_test_set.p03548 import problem_p03548


def test_problem_p03548_0():
    actual_output = problem_p03548("13 3 1")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03548_1():
    actual_output = problem_p03548("100000 1 1")
    expected_output = "49999"
    assert str(actual_output) == expected_output


def test_problem_p03548_2():
    actual_output = problem_p03548("13 3 1")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03548_3():
    actual_output = problem_p03548("64145 123 456")
    expected_output = "109"
    assert str(actual_output) == expected_output


def test_problem_p03548_4():
    actual_output = problem_p03548("12 3 1")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03548_5():
    actual_output = problem_p03548("64146 123 456")
    expected_output = "110"
    assert str(actual_output) == expected_output
