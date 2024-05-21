from pie_test_set.p03213 import problem_p03213


def test_problem_p03213_0():
    actual_output = problem_p03213("9")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03213_1():
    actual_output = problem_p03213("9")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03213_2():
    actual_output = problem_p03213("100")
    expected_output = "543"
    assert str(actual_output) == expected_output


def test_problem_p03213_3():
    actual_output = problem_p03213("10")
    expected_output = "1"
    assert str(actual_output) == expected_output
