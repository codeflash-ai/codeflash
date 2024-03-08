from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03310_0():
    input_content = "5\n3 2 4 1 2"
    expected_output = "2"
    run_pie_test_case("../p03310.py", input_content, expected_output)
