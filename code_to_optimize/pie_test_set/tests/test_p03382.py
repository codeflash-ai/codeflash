from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03382_0():
    input_content = "5\n6 9 4 2 11"
    expected_output = "11 6"
    run_pie_test_case("../p03382.py", input_content, expected_output)
