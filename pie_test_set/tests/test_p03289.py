from pie_test_set.p03289 import problem_p03289


def test_problem_p03289_0():
    actual_output = problem_p03289("AtCoder")
    expected_output = "AC"
    assert str(actual_output) == expected_output


def test_problem_p03289_1():
    actual_output = problem_p03289("AtCoCo")
    expected_output = "WA"
    assert str(actual_output) == expected_output


def test_problem_p03289_2():
    actual_output = problem_p03289("Atcoder")
    expected_output = "WA"
    assert str(actual_output) == expected_output


def test_problem_p03289_3():
    actual_output = problem_p03289("ACoder")
    expected_output = "WA"
    assert str(actual_output) == expected_output


def test_problem_p03289_4():
    actual_output = problem_p03289("AcycliC")
    expected_output = "WA"
    assert str(actual_output) == expected_output


def test_problem_p03289_5():
    actual_output = problem_p03289("AtCoder")
    expected_output = "AC"
    assert str(actual_output) == expected_output
