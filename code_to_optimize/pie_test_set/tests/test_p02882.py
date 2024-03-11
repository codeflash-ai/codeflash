from code_to_optimize.pie_test_set.p02882 import problem_p02882


def test_problem_p02882_0():
    actual_output = problem_p02882("2 2 4")
    expected_output = "45.0000000000"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02882_1():
    actual_output = problem_p02882("2 2 4")
    expected_output = "45.0000000000"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02882_2():
    actual_output = problem_p02882("12 21 10")
    expected_output = "89.7834636934"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02882_3():
    actual_output = problem_p02882("3 1 8")
    expected_output = "4.2363947991"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
