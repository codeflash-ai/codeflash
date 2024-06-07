from pie_test_set.p03264 import problem_p03264


def test_problem_p03264_0():
    actual_output = problem_p03264("3")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03264_1():
    actual_output = problem_p03264("6")
    expected_output = "9"
    assert str(actual_output) == expected_output


def test_problem_p03264_2():
    actual_output = problem_p03264("50")
    expected_output = "625"
    assert str(actual_output) == expected_output


def test_problem_p03264_3():
    actual_output = problem_p03264("11")
    expected_output = "30"
    assert str(actual_output) == expected_output


def test_problem_p03264_4():
    actual_output = problem_p03264("3")
    expected_output = "2"
    assert str(actual_output) == expected_output
