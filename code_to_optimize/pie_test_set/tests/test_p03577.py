from code_to_optimize.pie_test_set.p03577 import problem_p03577


def test_problem_p03577_0():
    actual_output = problem_p03577("CODEFESTIVAL")
    expected_output = "CODE"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03577_1():
    actual_output = problem_p03577("YAKINIKUFESTIVAL")
    expected_output = "YAKINIKU"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03577_2():
    actual_output = problem_p03577("CODEFESTIVALFESTIVAL")
    expected_output = "CODEFESTIVAL"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03577_3():
    actual_output = problem_p03577("CODEFESTIVAL")
    expected_output = "CODE"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
