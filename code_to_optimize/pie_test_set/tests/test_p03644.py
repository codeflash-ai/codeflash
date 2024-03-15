from code_to_optimize.pie_test_set.p03644 import problem_p03644


def test_problem_p03644_0():
    actual_output = problem_p03644("7")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03644_1():
    actual_output = problem_p03644("100")
    expected_output = "64"
    assert str(actual_output) == expected_output


def test_problem_p03644_2():
    actual_output = problem_p03644("1")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03644_3():
    actual_output = problem_p03644("7")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03644_4():
    actual_output = problem_p03644("32")
    expected_output = "32"
    assert str(actual_output) == expected_output
