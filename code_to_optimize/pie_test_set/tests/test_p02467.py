from code_to_optimize.pie_test_set.p02467 import problem_p02467


def test_problem_p02467_0():
    actual_output = problem_p02467("12")
    expected_output = "12: 2 2 3"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02467_1():
    actual_output = problem_p02467("12")
    expected_output = "12: 2 2 3"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02467_2():
    actual_output = problem_p02467("126")
    expected_output = "126: 2 3 3 7"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
