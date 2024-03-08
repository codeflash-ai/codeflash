from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00131_0():
    input_content = "1\n0 1 0 0 0 0 0 0 0 0\n1 1 1 0 0 0 0 0 0 0\n0 1 0 0 0 0 0 0 0 0\n0 0 0 0 1 1 0 0 0 0\n0 0 0 1 0 0 1 0 0 0\n0 0 0 0 1 1 0 0 0 0\n0 0 0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 1 0\n0 0 0 0 0 0 0 1 1 1\n0 0 0 0 0 0 0 0 1 0"
    expected_output = "0 0 0 0 0 0 0 0 0 0\n0 1 0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 0 0\n0 0 0 0 1 1 0 0 0 0\n0 0 0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 1 0\n0 0 0 0 0 0 0 0 0 0"
    run_pie_test_case("../p00131.py", input_content, expected_output)


def test_problem_p00131_1():
    input_content = "1\n0 1 0 0 0 0 0 0 0 0\n1 1 1 0 0 0 0 0 0 0\n0 1 0 0 0 0 0 0 0 0\n0 0 0 0 1 1 0 0 0 0\n0 0 0 1 0 0 1 0 0 0\n0 0 0 0 1 1 0 0 0 0\n0 0 0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 1 0\n0 0 0 0 0 0 0 1 1 1\n0 0 0 0 0 0 0 0 1 0"
    expected_output = "0 0 0 0 0 0 0 0 0 0\n0 1 0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 0 0\n0 0 0 0 1 1 0 0 0 0\n0 0 0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 1 0\n0 0 0 0 0 0 0 0 0 0"
    run_pie_test_case("../p00131.py", input_content, expected_output)
