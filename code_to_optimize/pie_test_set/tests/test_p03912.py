from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03912_0():
    input_content = "7 5\n3 1 4 1 5 9 2"
    expected_output = "3"
    run_pie_test_case("../p03912.py", input_content, expected_output)
