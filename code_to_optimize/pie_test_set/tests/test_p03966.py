from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03966_0():
    input_content = "3\n2 3\n1 1\n3 2"
    expected_output = "10"
    run_pie_test_case("../p03966.py", input_content, expected_output)
