from pie_test_set.p03577 import problem_p03577


def test_problem_p03577_0():
    actual_output = problem_p03577("CODEFESTIVAL")
    expected_output = "CODE"
    assert str(actual_output) == expected_output


def test_problem_p03577_1():
    actual_output = problem_p03577("YAKINIKUFESTIVAL")
    expected_output = "YAKINIKU"
    assert str(actual_output) == expected_output


def test_problem_p03577_2():
    actual_output = problem_p03577("CODEFESTIVALFESTIVAL")
    expected_output = "CODEFESTIVAL"
    assert str(actual_output) == expected_output


def test_problem_p03577_3():
    actual_output = problem_p03577("CODEFESTIVAL")
    expected_output = "CODE"
    assert str(actual_output) == expected_output
