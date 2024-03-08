from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00097_0():
    input_content = "3 6\n3 1\n0 0"
    expected_output = "3\n0"
    run_pie_test_case("../p00097.py", input_content, expected_output)


def test_problem_p00097_1():
    input_content = "3 6\n3 1\n0 0"
    expected_output = "3\n0"
    run_pie_test_case("../p00097.py", input_content, expected_output)
