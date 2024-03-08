from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00109_0():
    input_content = "2\n4-2*3=\n4*(8+4+3)="
    expected_output = "-2\n60"
    run_pie_test_case("../p00109.py", input_content, expected_output)


def test_problem_p00109_1():
    input_content = "2\n4-2*3=\n4*(8+4+3)="
    expected_output = "-2\n60"
    run_pie_test_case("../p00109.py", input_content, expected_output)
