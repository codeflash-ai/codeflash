from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03602_0():
    input_content = "3\n0 1 3\n1 0 2\n3 2 0"
    expected_output = "3"
    run_pie_test_case("../p03602.py", input_content, expected_output)
