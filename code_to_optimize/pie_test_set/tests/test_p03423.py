from code_to_optimize.pie_test_set.p03423 import problem_p03423


def test_problem_p03423_0():
    actual_output = problem_p03423("8")
    expected_output = "2"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
