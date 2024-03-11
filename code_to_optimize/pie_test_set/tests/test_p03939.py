from code_to_optimize.pie_test_set.p03939 import problem_p03939


def test_problem_p03939_0():
    actual_output = problem_p03939("1 3 3")
    expected_output = "4.500000000000000"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03939_1():
    actual_output = problem_p03939("1 3 3")
    expected_output = "4.500000000000000"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03939_2():
    actual_output = problem_p03939("1000 100 100")
    expected_output = "649620280.957660079002380"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03939_3():
    actual_output = problem_p03939("2 1 0")
    expected_output = "2.500000000000000"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
