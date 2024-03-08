from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03495_0():
    input_content = "5 2\n1 1 2 2 5"
    expected_output = "1"
    run_pie_test_case("../p03495.py", input_content, expected_output)
