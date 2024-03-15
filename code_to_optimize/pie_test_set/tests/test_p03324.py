from code_to_optimize.pie_test_set.p03324 import problem_p03324


def test_problem_p03324_0():
    actual_output = problem_p03324("0 5")
    expected_output = "5"
    assert str(actual_output) == expected_output


def test_problem_p03324_1():
    actual_output = problem_p03324("2 85")
    expected_output = "850000"
    assert str(actual_output) == expected_output


def test_problem_p03324_2():
    actual_output = problem_p03324("1 11")
    expected_output = "1100"
    assert str(actual_output) == expected_output


def test_problem_p03324_3():
    actual_output = problem_p03324("0 5")
    expected_output = "5"
    assert str(actual_output) == expected_output
