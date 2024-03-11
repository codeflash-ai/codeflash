from code_to_optimize.pie_test_set.p02731 import problem_p02731


def test_problem_p02731_0():
    actual_output = problem_p02731("3")
    expected_output = "1.000000000000"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02731_1():
    actual_output = problem_p02731("999")
    expected_output = "36926037.000000000000"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02731_2():
    actual_output = problem_p02731("3")
    expected_output = "1.000000000000"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
