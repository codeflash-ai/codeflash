from pie_test_set.p03777 import problem_p03777


def test_problem_p03777_0():
    actual_output = problem_p03777("H H")
    expected_output = "H"
    assert str(actual_output) == expected_output


def test_problem_p03777_1():
    actual_output = problem_p03777("D D")
    expected_output = "H"
    assert str(actual_output) == expected_output


def test_problem_p03777_2():
    actual_output = problem_p03777("H H")
    expected_output = "H"
    assert str(actual_output) == expected_output


def test_problem_p03777_3():
    actual_output = problem_p03777("D H")
    expected_output = "D"
    assert str(actual_output) == expected_output
