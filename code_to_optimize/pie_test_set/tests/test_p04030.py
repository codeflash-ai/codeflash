from code_to_optimize.pie_test_set.p04030 import problem_p04030


def test_problem_p04030_0():
    actual_output = problem_p04030("01B0")
    expected_output = "00"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p04030_1():
    actual_output = problem_p04030("01B0")
    expected_output = "00"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p04030_2():
    actual_output = problem_p04030("0BB1")
    expected_output = "1"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
