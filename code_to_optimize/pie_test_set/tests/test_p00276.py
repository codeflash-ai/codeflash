from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00276_0():
    input_content = "4\n3 0 0\n1 1 1\n9 4 1\n0 1 2"
    expected_output = "1\n1\n4\n0"
    run_pie_test_case("../p00276.py", input_content, expected_output)


def test_problem_p00276_1():
    input_content = "4\n3 0 0\n1 1 1\n9 4 1\n0 1 2"
    expected_output = "1\n1\n4\n0"
    run_pie_test_case("../p00276.py", input_content, expected_output)
