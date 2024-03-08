from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03817_0():
    input_content = "7"
    expected_output = "2"
    run_pie_test_case("../p03817.py", input_content, expected_output)
