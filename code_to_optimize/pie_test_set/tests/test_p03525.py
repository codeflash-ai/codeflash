from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03525_0():
    input_content = "3\n7 12 8"
    expected_output = "4"
    run_pie_test_case("../p03525.py", input_content, expected_output)
