from code_to_optimize.pie_test_set.p03437 import problem_p03437


def test_problem_p03437_0():
    actual_output = problem_p03437("8 6")
    expected_output = "16"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03437_1():
    actual_output = problem_p03437("3 3")
    expected_output = "-1"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03437_2():
    actual_output = problem_p03437("8 6")
    expected_output = "16"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
