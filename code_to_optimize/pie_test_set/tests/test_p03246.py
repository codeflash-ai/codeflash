from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03246_0():
    input_content = "4\n3 1 3 2"
    expected_output = "1"
    run_pie_test_case("../p03246.py", input_content, expected_output)
