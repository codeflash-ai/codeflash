from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00020_0():
    input_content = "this is a pen."
    expected_output = "THIS IS A PEN."
    run_pie_test_case("../p00020.py", input_content, expected_output)


def test_problem_p00020_1():
    input_content = "this is a pen."
    expected_output = "THIS IS A PEN."
    run_pie_test_case("../p00020.py", input_content, expected_output)
