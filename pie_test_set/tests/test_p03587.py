from pie_test_set.p03587 import problem_p03587


def test_problem_p03587_0():
    actual_output = problem_p03587("111100")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03587_1():
    actual_output = problem_p03587("001001")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03587_2():
    actual_output = problem_p03587("000000")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03587_3():
    actual_output = problem_p03587("111100")
    expected_output = "4"
    assert str(actual_output) == expected_output
