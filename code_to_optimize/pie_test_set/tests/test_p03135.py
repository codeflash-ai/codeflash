from code_to_optimize.pie_test_set.p03135 import problem_p03135


def test_problem_p03135_0():
    actual_output = problem_p03135("8 3")
    expected_output = "2.6666666667"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03135_1():
    actual_output = problem_p03135("1 100")
    expected_output = "0.0100000000"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03135_2():
    actual_output = problem_p03135("99 1")
    expected_output = "99.0000000000"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03135_3():
    actual_output = problem_p03135("8 3")
    expected_output = "2.6666666667"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
