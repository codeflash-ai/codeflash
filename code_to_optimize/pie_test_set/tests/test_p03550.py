from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03550_0():
    input_content = "3 100 100\n10 1000 100"
    expected_output = "900"
    run_pie_test_case("../p03550.py", input_content, expected_output)
