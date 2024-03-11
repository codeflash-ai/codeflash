from code_to_optimize.pie_test_set.p03304 import problem_p03304


def test_problem_p03304_0():
    actual_output = problem_p03304("2 3 1")
    expected_output = "1.0000000000"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03304_1():
    actual_output = problem_p03304("2 3 1")
    expected_output = "1.0000000000"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03304_2():
    actual_output = problem_p03304("1000000000 180707 0")
    expected_output = "0.0001807060"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
