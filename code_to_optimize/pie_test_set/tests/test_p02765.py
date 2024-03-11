from code_to_optimize.pie_test_set.p02765 import problem_p02765


def test_problem_p02765_0():
    actual_output = problem_p02765("2 2919")
    expected_output = "3719"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02765_1():
    actual_output = problem_p02765("2 2919")
    expected_output = "3719"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02765_2():
    actual_output = problem_p02765("22 3051")
    expected_output = "3051"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
