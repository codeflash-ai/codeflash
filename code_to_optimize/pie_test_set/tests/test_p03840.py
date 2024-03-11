from code_to_optimize.pie_test_set.p03840 import problem_p03840


def test_problem_p03840_0():
    actual_output = problem_p03840("2 1 1 0 0 0 0")
    expected_output = "3"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03840_1():
    actual_output = problem_p03840("2 1 1 0 0 0 0")
    expected_output = "3"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03840_2():
    actual_output = problem_p03840("0 0 10 0 0 0 0")
    expected_output = "0"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
