from code_to_optimize.pie_test_set.p03671 import problem_p03671


def test_problem_p03671_0():
    actual_output = problem_p03671("700 600 780")
    expected_output = "1300"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03671_1():
    actual_output = problem_p03671("10000 10000 10000")
    expected_output = "20000"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03671_2():
    actual_output = problem_p03671("700 600 780")
    expected_output = "1300"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
