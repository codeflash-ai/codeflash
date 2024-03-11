from code_to_optimize.pie_test_set.p02954 import problem_p02954


def test_problem_p02954_0():
    actual_output = problem_p02954("RRLRL")
    expected_output = "0 1 2 1 1"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02954_1():
    actual_output = problem_p02954("RRLLLLRLRRLL")
    expected_output = "0 3 3 0 0 0 1 1 0 2 2 0"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02954_2():
    actual_output = problem_p02954("RRRLLRLLRRRLLLLL")
    expected_output = "0 0 3 2 0 2 1 0 0 0 4 4 0 0 0 0"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02954_3():
    actual_output = problem_p02954("RRLRL")
    expected_output = "0 1 2 1 1"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
