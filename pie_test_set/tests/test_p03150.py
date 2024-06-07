from pie_test_set.p03150 import problem_p03150


def test_problem_p03150_0():
    actual_output = problem_p03150("keyofscience")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03150_1():
    actual_output = problem_p03150("keyence")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03150_2():
    actual_output = problem_p03150("ashlfyha")
    expected_output = "NO"
    assert str(actual_output) == expected_output


def test_problem_p03150_3():
    actual_output = problem_p03150("keyofscience")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03150_4():
    actual_output = problem_p03150("mpyszsbznf")
    expected_output = "NO"
    assert str(actual_output) == expected_output
