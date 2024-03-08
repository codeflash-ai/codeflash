from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00354_0():
    input_content = "1"
    expected_output = "fri"
    run_pie_test_case("../p00354.py", input_content, expected_output)
