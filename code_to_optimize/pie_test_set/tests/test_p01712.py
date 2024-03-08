from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01712_0():
    input_content = "3 9 9\n2 2 2\n5 5 2\n8 8 2"
    expected_output = "Yes"
    run_pie_test_case("../p01712.py", input_content, expected_output)
