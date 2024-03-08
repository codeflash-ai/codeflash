from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03716_0():
    input_content = "2\n3 1 4 1 5 9"
    expected_output = "1"
    run_pie_test_case("../p03716.py", input_content, expected_output)
