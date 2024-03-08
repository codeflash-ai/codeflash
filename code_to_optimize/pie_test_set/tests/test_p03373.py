from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03373_0():
    input_content = "1500 2000 1600 3 2"
    expected_output = "7900"
    run_pie_test_case("../p03373.py", input_content, expected_output)
