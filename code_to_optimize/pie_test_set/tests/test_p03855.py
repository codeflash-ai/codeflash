from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03855_0():
    input_content = "4 3 1\n1 2\n2 3\n3 4\n2 3"
    expected_output = "1 2 2 1"
    run_pie_test_case("../p03855.py", input_content, expected_output)
