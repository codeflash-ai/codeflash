from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00526_0():
    input_content = "10\n1 1 0 0 1 0 1 1 1 0"
    expected_output = "7"
    run_pie_test_case("../p00526.py", input_content, expected_output)


def test_problem_p00526_1():
    input_content = "10\n1 1 0 0 1 0 1 1 1 0"
    expected_output = "7"
    run_pie_test_case("../p00526.py", input_content, expected_output)
