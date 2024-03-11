from code_to_optimize.pie_test_set.p02939 import problem_p02939


def test_problem_p02939_0():
    actual_output = problem_p02939("aabbaa")
    expected_output = "4"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02939_1():
    actual_output = problem_p02939("aaaccacabaababc")
    expected_output = "12"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02939_2():
    actual_output = problem_p02939("aabbaa")
    expected_output = "4"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
