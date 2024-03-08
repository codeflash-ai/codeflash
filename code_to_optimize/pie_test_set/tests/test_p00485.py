from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00485_0():
    input_content = "3 3 1\n1 2 1\n2 3 1\n3 1 1\n1"
    expected_output = "2"
    run_pie_test_case("../p00485.py", input_content, expected_output)


def test_problem_p00485_1():
    input_content = "3 3 1\n1 2 1\n2 3 1\n3 1 1\n1"
    expected_output = "2"
    run_pie_test_case("../p00485.py", input_content, expected_output)
