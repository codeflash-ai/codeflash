from pie_test_set.p03268 import problem_p03268


def test_problem_p03268_0():
    actual_output = problem_p03268("3 2")
    expected_output = "9"
    assert str(actual_output) == expected_output


def test_problem_p03268_1():
    actual_output = problem_p03268("31415 9265")
    expected_output = "27"
    assert str(actual_output) == expected_output


def test_problem_p03268_2():
    actual_output = problem_p03268("3 2")
    expected_output = "9"
    assert str(actual_output) == expected_output


def test_problem_p03268_3():
    actual_output = problem_p03268("35897 932")
    expected_output = "114191"
    assert str(actual_output) == expected_output


def test_problem_p03268_4():
    actual_output = problem_p03268("5 3")
    expected_output = "1"
    assert str(actual_output) == expected_output
