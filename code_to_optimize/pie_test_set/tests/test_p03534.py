from code_to_optimize.pie_test_set.p03534 import problem_p03534


def test_problem_p03534_0():
    actual_output = problem_p03534("abac")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03534_1():
    actual_output = problem_p03534("aba")
    expected_output = "NO"
    assert str(actual_output) == expected_output


def test_problem_p03534_2():
    actual_output = problem_p03534("abac")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03534_3():
    actual_output = problem_p03534("babacccabab")
    expected_output = "YES"
    assert str(actual_output) == expected_output
