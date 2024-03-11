from code_to_optimize.pie_test_set.p02639 import problem_p02639


def test_problem_p02639_0():
    actual_output = problem_p02639("0 2 3 4 5")
    expected_output = "1"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
