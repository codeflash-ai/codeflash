from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03682_0():
    input_content = "3\n1 5\n3 9\n7 8"
    expected_output = "3"
    run_pie_test_case("../p03682.py", input_content, expected_output)
