from code_to_optimize.pie_test_set.p03206 import problem_p03206


def test_problem_p03206_0():
    actual_output = problem_p03206("25")
    expected_output = "Christmas"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03206_1():
    actual_output = problem_p03206("25")
    expected_output = "Christmas"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03206_2():
    actual_output = problem_p03206("22")
    expected_output = "Christmas Eve Eve Eve"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
