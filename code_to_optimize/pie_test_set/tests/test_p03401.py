from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03401_0():
    input_content = "3\n3 5 -1"
    expected_output = "12\n8\n10"
    run_pie_test_case("../p03401.py", input_content, expected_output)
