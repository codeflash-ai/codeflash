from code_to_optimize.pie_test_set.p03291 import problem_p03291


def test_problem_p03291_0():
    actual_output = problem_p03291("A??C")
    expected_output = "8"
    assert str(actual_output) == expected_output


def test_problem_p03291_1():
    actual_output = problem_p03291("ABCBC")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03291_2():
    actual_output = problem_p03291("????C?????B??????A???????")
    expected_output = "979596887"
    assert str(actual_output) == expected_output


def test_problem_p03291_3():
    actual_output = problem_p03291("A??C")
    expected_output = "8"
    assert str(actual_output) == expected_output
