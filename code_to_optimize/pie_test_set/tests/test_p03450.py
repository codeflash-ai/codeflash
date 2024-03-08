from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03450_0():
    input_content = "3 3\n1 2 1\n2 3 1\n1 3 2"
    expected_output = "Yes"
    run_pie_test_case("../p03450.py", input_content, expected_output)
