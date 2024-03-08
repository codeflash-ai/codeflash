from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03266_0():
    input_content = "3 2"
    expected_output = "9"
    run_pie_test_case("../p03266.py", input_content, expected_output)
