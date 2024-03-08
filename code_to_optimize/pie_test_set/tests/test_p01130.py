from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01130_0():
    input_content = "4 5 1 3 4\n1 2 5\n2 3 5\n2 4 5\n1 3 8\n1 4 8\n0 0 0 0 0"
    expected_output = "15"
    run_pie_test_case("../p01130.py", input_content, expected_output)


def test_problem_p01130_1():
    input_content = "4 5 1 3 4\n1 2 5\n2 3 5\n2 4 5\n1 3 8\n1 4 8\n0 0 0 0 0"
    expected_output = "15"
    run_pie_test_case("../p01130.py", input_content, expected_output)
