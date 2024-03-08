from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00472_0():
    input_content = "7 5\n2\n1\n1\n3\n2\n1\n2\n-1\n3\n2\n-3"
    expected_output = "18"
    run_pie_test_case("../p00472.py", input_content, expected_output)


def test_problem_p00472_1():
    input_content = "7 5\n2\n1\n1\n3\n2\n1\n2\n-1\n3\n2\n-3"
    expected_output = "18"
    run_pie_test_case("../p00472.py", input_content, expected_output)
