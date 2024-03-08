from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03885_0():
    input_content = "2\n0 1\n1 0"
    expected_output = "6"
    run_pie_test_case("../p03885.py", input_content, expected_output)
