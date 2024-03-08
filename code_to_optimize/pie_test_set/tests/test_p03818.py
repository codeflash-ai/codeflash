from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03818_0():
    input_content = "5\n1 2 1 3 7"
    expected_output = "3"
    run_pie_test_case("../p03818.py", input_content, expected_output)
