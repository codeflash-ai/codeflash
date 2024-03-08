from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03639_0():
    input_content = "3\n1 10 100"
    expected_output = "Yes"
    run_pie_test_case("../p03639.py", input_content, expected_output)
