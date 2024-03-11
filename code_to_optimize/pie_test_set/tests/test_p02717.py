from code_to_optimize.pie_test_set.p02717 import problem_p02717


def test_problem_p02717_0():
    actual_output = problem_p02717("1 2 3")
    expected_output = "3 1 2"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02717_1():
    actual_output = problem_p02717("100 100 100")
    expected_output = "100 100 100"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02717_2():
    actual_output = problem_p02717("41 59 31")
    expected_output = "31 41 59"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02717_3():
    actual_output = problem_p02717("1 2 3")
    expected_output = "3 1 2"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
