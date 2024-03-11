from code_to_optimize.pie_test_set.p03657 import problem_p03657


def test_problem_p03657_0():
    actual_output = problem_p03657("4 5")
    expected_output = "Possible"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03657_1():
    actual_output = problem_p03657("4 5")
    expected_output = "Possible"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03657_2():
    actual_output = problem_p03657("1 1")
    expected_output = "Impossible"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
