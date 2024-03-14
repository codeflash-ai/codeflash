from code_to_optimize.pie_test_set.p03222 import problem_p03222


def test_problem_p03222_0():
    actual_output = problem_p03222("1 3 2")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03222_1():
    actual_output = problem_p03222("2 3 1")
    expected_output = "5"
    assert str(actual_output) == expected_output


def test_problem_p03222_2():
    actual_output = problem_p03222("1 3 2")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03222_3():
    actual_output = problem_p03222("1 3 1")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03222_4():
    actual_output = problem_p03222("7 1 1")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03222_5():
    actual_output = problem_p03222("2 3 3")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03222_6():
    actual_output = problem_p03222("15 8 5")
    expected_output = "437760187"
    assert str(actual_output) == expected_output
