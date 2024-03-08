from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00502_0():
    input_content = "3 4\n31\n27\n35\n20 25 30\n23 29 90\n21 35 60\n28 33 40"
    expected_output = "80"
    run_pie_test_case("../p00502.py", input_content, expected_output)


def test_problem_p00502_1():
    input_content = "3 4\n31\n27\n35\n20 25 30\n23 29 90\n21 35 60\n28 33 40"
    expected_output = "80"
    run_pie_test_case("../p00502.py", input_content, expected_output)
