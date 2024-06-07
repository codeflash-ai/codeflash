from pie_test_set.p02702 import problem_p02702


def test_problem_p02702_0():
    actual_output = problem_p02702("1817181712114")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02702_1():
    actual_output = problem_p02702("1817181712114")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02702_2():
    actual_output = problem_p02702("2119")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p02702_3():
    actual_output = problem_p02702("14282668646")
    expected_output = "2"
    assert str(actual_output) == expected_output
