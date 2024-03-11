from code_to_optimize.pie_test_set.p02576 import problem_p02576


def test_problem_p02576_0():
    actual_output = problem_p02576("20 12 6")
    expected_output = "12"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02576_1():
    actual_output = problem_p02576("1000 1 1000")
    expected_output = "1000000"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02576_2():
    actual_output = problem_p02576("20 12 6")
    expected_output = "12"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
