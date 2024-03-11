from code_to_optimize.pie_test_set.p03623 import problem_p03623


def test_problem_p03623_0():
    actual_output = problem_p03623("5 2 7")
    expected_output = "B"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03623_1():
    actual_output = problem_p03623("1 999 1000")
    expected_output = "A"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03623_2():
    actual_output = problem_p03623("5 2 7")
    expected_output = "B"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
