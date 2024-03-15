from code_to_optimize.pie_test_set.p03729 import problem_p03729


def test_problem_p03729_0():
    actual_output = problem_p03729("rng gorilla apple")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03729_1():
    actual_output = problem_p03729("a a a")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03729_2():
    actual_output = problem_p03729("rng gorilla apple")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03729_3():
    actual_output = problem_p03729("aaaaaaaaab aaaaaaaaaa aaaaaaaaab")
    expected_output = "NO"
    assert str(actual_output) == expected_output


def test_problem_p03729_4():
    actual_output = problem_p03729("yakiniku unagi sushi")
    expected_output = "NO"
    assert str(actual_output) == expected_output
