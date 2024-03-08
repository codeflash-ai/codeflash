from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00730_0():
    input_content = "3 5 6\n1 18\n2 19\n1 2\n3 4 1\n1 1\n2 1\n3 1\n0 2 5\n0 0 0"
    expected_output = "4 4 6 16\n1 1 1 1\n10"
    run_pie_test_case("../p00730.py", input_content, expected_output)


def test_problem_p00730_1():
    input_content = "3 5 6\n1 18\n2 19\n1 2\n3 4 1\n1 1\n2 1\n3 1\n0 2 5\n0 0 0"
    expected_output = "4 4 6 16\n1 1 1 1\n10"
    run_pie_test_case("../p00730.py", input_content, expected_output)
