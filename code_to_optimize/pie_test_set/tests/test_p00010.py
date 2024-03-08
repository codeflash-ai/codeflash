from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00010_0():
    input_content = "1\n0.0 0.0 2.0 0.0 2.0 2.0"
    expected_output = "1.000 1.000 1.414"
    run_pie_test_case("../p00010.py", input_content, expected_output)


def test_problem_p00010_1():
    input_content = "1\n0.0 0.0 2.0 0.0 2.0 2.0"
    expected_output = "1.000 1.000 1.414"
    run_pie_test_case("../p00010.py", input_content, expected_output)
