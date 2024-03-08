from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03423_0():
    input_content = "8"
    expected_output = "2"
    run_pie_test_case("../p03423.py", input_content, expected_output)
