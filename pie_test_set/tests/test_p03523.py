from pie_test_set.p03523 import problem_p03523


def test_problem_p03523_0():
    actual_output = problem_p03523("KIHBR")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03523_1():
    actual_output = problem_p03523("AKIBAHARA")
    expected_output = "NO"
    assert str(actual_output) == expected_output


def test_problem_p03523_2():
    actual_output = problem_p03523("KIHBR")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03523_3():
    actual_output = problem_p03523("AAKIAHBAARA")
    expected_output = "NO"
    assert str(actual_output) == expected_output
