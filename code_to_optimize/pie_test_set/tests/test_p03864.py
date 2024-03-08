from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03864_0():
    input_content = "3 3\n2 2 2"
    expected_output = "1"
    run_pie_test_case("../p03864.py", input_content, expected_output)
