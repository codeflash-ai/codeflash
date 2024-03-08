from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03459_0():
    input_content = "2\n3 1 2\n6 1 1"
    expected_output = "Yes"
    run_pie_test_case("../p03459.py", input_content, expected_output)
