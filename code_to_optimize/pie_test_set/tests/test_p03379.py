from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03379_0():
    input_content = "4\n2 4 4 3"
    expected_output = "4\n3\n3\n4"
    run_pie_test_case("../p03379.py", input_content, expected_output)
