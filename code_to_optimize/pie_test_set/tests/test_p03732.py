from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03732_0():
    input_content = "4 6\n2 1\n3 4\n4 10\n3 4"
    expected_output = "11"
    run_pie_test_case("../p03732.py", input_content, expected_output)
