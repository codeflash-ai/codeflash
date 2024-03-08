from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03673_0():
    input_content = "4\n1 2 3 4"
    expected_output = "4 2 1 3"
    run_pie_test_case("../p03673.py", input_content, expected_output)
