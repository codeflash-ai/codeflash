from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03645_0():
    input_content = "3 2\n1 2\n2 3"
    expected_output = "POSSIBLE"
    run_pie_test_case("../p03645.py", input_content, expected_output)
