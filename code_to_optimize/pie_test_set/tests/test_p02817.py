from code_to_optimize.pie_test_set.p02817 import problem_p02817


def test_problem_p02817_0():
    actual_output = problem_p02817("oder atc")
    expected_output = "atcoder"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02817_1():
    actual_output = problem_p02817("humu humu")
    expected_output = "humuhumu"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02817_2():
    actual_output = problem_p02817("oder atc")
    expected_output = "atcoder"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
