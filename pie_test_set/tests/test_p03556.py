from pie_test_set.p03556 import problem_p03556


def test_problem_p03556_0():
    actual_output = problem_p03556("10")
    expected_output = "9"
    assert str(actual_output) == expected_output


def test_problem_p03556_1():
    actual_output = problem_p03556("81")
    expected_output = "81"
    assert str(actual_output) == expected_output


def test_problem_p03556_2():
    actual_output = problem_p03556("10")
    expected_output = "9"
    assert str(actual_output) == expected_output


def test_problem_p03556_3():
    actual_output = problem_p03556("271828182")
    expected_output = "271821169"
    assert str(actual_output) == expected_output
