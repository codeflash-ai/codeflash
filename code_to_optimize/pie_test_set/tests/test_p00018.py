from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00018_0():
    input_content = "3 6 9 7 5"
    expected_output = "9 7 6 5 3"
    run_pie_test_case("../p00018.py", input_content, expected_output)
