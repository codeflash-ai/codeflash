from pie_test_set.p02546 import problem_p02546


def test_problem_p02546_0():
    actual_output = problem_p02546("apple")
    expected_output = "apples"
    assert str(actual_output) == expected_output


def test_problem_p02546_1():
    actual_output = problem_p02546("box")
    expected_output = "boxs"
    assert str(actual_output) == expected_output


def test_problem_p02546_2():
    actual_output = problem_p02546("bus")
    expected_output = "buses"
    assert str(actual_output) == expected_output


def test_problem_p02546_3():
    actual_output = problem_p02546("apple")
    expected_output = "apples"
    assert str(actual_output) == expected_output
