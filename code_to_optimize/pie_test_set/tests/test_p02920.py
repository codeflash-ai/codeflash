from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02920_0():
    input_content = "2\n4 2 3 1"
    expected_output = "Yes"
    run_pie_test_case("../p02920.py", input_content, expected_output)
