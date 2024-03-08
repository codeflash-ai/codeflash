from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00145_0():
    input_content = "3\n3 5\n2 8\n5 4"
    expected_output = "440"
    run_pie_test_case("../p00145.py", input_content, expected_output)


def test_problem_p00145_1():
    input_content = "3\n3 5\n2 8\n5 4"
    expected_output = "440"
    run_pie_test_case("../p00145.py", input_content, expected_output)
