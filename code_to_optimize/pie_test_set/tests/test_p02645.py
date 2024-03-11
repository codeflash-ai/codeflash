from code_to_optimize.pie_test_set.p02645 import problem_p02645


def test_problem_p02645_0():
    actual_output = problem_p02645("takahashi")
    expected_output = "tak"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02645_1():
    actual_output = problem_p02645("naohiro")
    expected_output = "nao"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02645_2():
    actual_output = problem_p02645("takahashi")
    expected_output = "tak"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
