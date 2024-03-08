from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00516_0():
    input_content = "4 3\n5\n3\n1\n4\n4\n3\n2"
    expected_output = "2"
    run_pie_test_case("../p00516.py", input_content, expected_output)
