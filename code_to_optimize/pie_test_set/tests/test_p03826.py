from code_to_optimize.pie_test_set.p03826 import problem_p03826


def test_problem_p03826_0():
    actual_output = problem_p03826("3 5 2 7")
    expected_output = "15"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03826_1():
    actual_output = problem_p03826("100 600 200 300")
    expected_output = "60000"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03826_2():
    actual_output = problem_p03826("3 5 2 7")
    expected_output = "15"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
