from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00455_0():
    input_content = "9 0 0 18 0 0\n9 0 1 18 0 0\n12 14 52 12 15 30"
    expected_output = "9 0 0\n8 59 59\n0 0 38"
    run_pie_test_case("../p00455.py", input_content, expected_output)


def test_problem_p00455_1():
    input_content = "9 0 0 18 0 0\n9 0 1 18 0 0\n12 14 52 12 15 30"
    expected_output = "9 0 0\n8 59 59\n0 0 38"
    run_pie_test_case("../p00455.py", input_content, expected_output)


def test_problem_p00455_2():
    input_content = "None"
    expected_output = "None"
    run_pie_test_case("../p00455.py", input_content, expected_output)
