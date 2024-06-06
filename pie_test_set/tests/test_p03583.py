from pie_test_set.p03583 import problem_p03583


def test_problem_p03583_0():
    actual_output = problem_p03583("2")
    expected_output = "1 2 2"
    assert str(actual_output) == expected_output


def test_problem_p03583_1():
    actual_output = problem_p03583("2")
    expected_output = "1 2 2"
    assert str(actual_output) == expected_output


def test_problem_p03583_2():
    actual_output = problem_p03583("4664")
    expected_output = "3498 3498 3498"
    assert str(actual_output) == expected_output


def test_problem_p03583_3():
    actual_output = problem_p03583("3485")
    expected_output = "872 1012974 1539173474040"
    assert str(actual_output) == expected_output
