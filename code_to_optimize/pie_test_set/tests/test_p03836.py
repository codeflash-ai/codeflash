from code_to_optimize.pie_test_set.p03836 import problem_p03836


def test_problem_p03836_0():
    actual_output = problem_p03836("0 0 1 2")
    expected_output = "UURDDLLUUURRDRDDDLLU"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03836_1():
    actual_output = problem_p03836("0 0 1 2")
    expected_output = "UURDDLLUUURRDRDDDLLU"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03836_2():
    actual_output = problem_p03836("-2 -2 1 1")
    expected_output = "UURRURRDDDLLDLLULUUURRURRDDDLLDL"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
