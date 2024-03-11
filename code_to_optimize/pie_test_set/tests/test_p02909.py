from code_to_optimize.pie_test_set.p02909 import problem_p02909


def test_problem_p02909_0():
    actual_output = problem_p02909("Sunny")
    expected_output = "Cloudy"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02909_1():
    actual_output = problem_p02909("Rainy")
    expected_output = "Sunny"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02909_2():
    actual_output = problem_p02909("Sunny")
    expected_output = "Cloudy"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
