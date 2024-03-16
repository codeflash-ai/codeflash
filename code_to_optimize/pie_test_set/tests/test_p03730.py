from code_to_optimize.pie_test_set.p03730 import problem_p03730


def test_problem_p03730_0():
    actual_output = problem_p03730("7 5 1")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03730_1():
    actual_output = problem_p03730("7 5 1")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03730_2():
    actual_output = problem_p03730("2 2 1")
    expected_output = "NO"
    assert str(actual_output) == expected_output


def test_problem_p03730_3():
    actual_output = problem_p03730("77 42 36")
    expected_output = "NO"
    assert str(actual_output) == expected_output


def test_problem_p03730_4():
    actual_output = problem_p03730("40 98 58")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03730_5():
    actual_output = problem_p03730("1 100 97")
    expected_output = "YES"
    assert str(actual_output) == expected_output
