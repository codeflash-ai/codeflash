from code_to_optimize.pie_test_set.p02946 import problem_p02946


def test_problem_p02946_0():
    actual_output = problem_p02946("3 7")
    expected_output = "5 6 7 8 9"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02946_1():
    actual_output = problem_p02946("4 0")
    expected_output = "-3 -2 -1 0 1 2 3"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02946_2():
    actual_output = problem_p02946("1 100")
    expected_output = "100"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02946_3():
    actual_output = problem_p02946("3 7")
    expected_output = "5 6 7 8 9"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
