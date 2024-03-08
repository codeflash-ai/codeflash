from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03764_0():
    input_content = "3 3\n1 3 4\n1 3 6"
    expected_output = "60"
    run_pie_test_case("../p03764.py", input_content, expected_output)
