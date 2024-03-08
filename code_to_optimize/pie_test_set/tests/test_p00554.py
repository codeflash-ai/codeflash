from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00554_0():
    input_content = "4 5\n1 7\n6 2\n3 5\n4 4\n0 8"
    expected_output = "4"
    run_pie_test_case("../p00554.py", input_content, expected_output)
