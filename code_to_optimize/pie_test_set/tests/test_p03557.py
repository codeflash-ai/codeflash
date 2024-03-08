from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03557_0():
    input_content = "2\n1 5\n2 4\n3 6"
    expected_output = "3"
    run_pie_test_case("../p03557.py", input_content, expected_output)
