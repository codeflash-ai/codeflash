from code_to_optimize.pie_test_set.p04048 import problem_p04048


def test_problem_p04048_0():
    actual_output = problem_p04048("5 2")
    expected_output = "12"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p04048_1():
    actual_output = problem_p04048("5 2")
    expected_output = "12"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
