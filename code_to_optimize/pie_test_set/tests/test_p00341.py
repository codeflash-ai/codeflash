from code_to_optimize.pie_test_set.p00341 import problem_p00341


def test_problem_p00341_0():
    actual_output = problem_p00341("1 1 3 4 8 9 7 3 4 5 5 5")
    expected_output = "no"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p00341_1():
    actual_output = problem_p00341("1 1 2 2 3 1 2 3 3 3 1 2")
    expected_output = "yes"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p00341_2():
    actual_output = problem_p00341("1 1 3 4 8 9 7 3 4 5 5 5")
    expected_output = "no"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
