from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03741_0():
    input_content = "4\n1 -3 1 0"
    expected_output = "4"
    run_pie_test_case("../p03741.py", input_content, expected_output)
