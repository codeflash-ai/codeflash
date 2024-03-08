from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03223_0():
    input_content = "5\n6\n8\n1\n2\n3"
    expected_output = "21"
    run_pie_test_case("../p03223.py", input_content, expected_output)
