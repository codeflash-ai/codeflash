from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03536_0():
    input_content = "3\n0 2\n1 3\n3 4"
    expected_output = "2"
    run_pie_test_case("../p03536.py", input_content, expected_output)
