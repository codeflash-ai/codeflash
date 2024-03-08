from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00023_0():
    input_content = "2\n0.0 0.0 5.0 0.0 0.0 4.0\n0.0 0.0 2.0 4.1 0.0 2.0"
    expected_output = "2\n0"
    run_pie_test_case("../p00023.py", input_content, expected_output)


def test_problem_p00023_1():
    input_content = "2\n0.0 0.0 5.0 0.0 0.0 4.0\n0.0 0.0 2.0 4.1 0.0 2.0"
    expected_output = "2\n0"
    run_pie_test_case("../p00023.py", input_content, expected_output)
