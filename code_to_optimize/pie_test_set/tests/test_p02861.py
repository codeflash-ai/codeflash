from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02861_0():
    input_content = "3\n0 0\n1 0\n0 1"
    expected_output = "2.2761423749"
    run_pie_test_case("../p02861.py", input_content, expected_output)
