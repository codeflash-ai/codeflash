from code_to_optimize.pie_test_set.p03332 import problem_p03332


def test_problem_p03332_0():
    actual_output = problem_p03332("4 1 2 5")
    expected_output = "40"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03332_1():
    actual_output = problem_p03332("90081 33447 90629 6391049189")
    expected_output = "577742975"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03332_2():
    actual_output = problem_p03332("2 5 6 0")
    expected_output = "1"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03332_3():
    actual_output = problem_p03332("4 1 2 5")
    expected_output = "40"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
