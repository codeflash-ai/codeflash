from code_to_optimize.pie_test_set.p00322 import problem_p00322


def test_problem_p00322_0():
    actual_output = problem_p00322("7 6 -1 1 -1 9 2 3 4")
    expected_output = "1"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p00322_1():
    actual_output = problem_p00322("7 6 5 1 8 9 2 3 4")
    expected_output = "0"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p00322_2():
    actual_output = problem_p00322("-1 -1 -1 -1 -1 -1 8 4 6")
    expected_output = "12"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p00322_3():
    actual_output = problem_p00322("7 6 -1 1 -1 9 2 3 4")
    expected_output = "1"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p00322_4():
    actual_output = problem_p00322("-1 -1 -1 -1 -1 -1 -1 -1 -1")
    expected_output = "168"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
