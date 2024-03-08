from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00508_0():
    input_content = "2\n0 0\n1 1"
    expected_output = "2"
    run_pie_test_case("../p00508.py", input_content, expected_output)
