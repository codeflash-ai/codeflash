from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03967_0():
    input_content = "gpg"
    expected_output = "0"
    run_pie_test_case("../p03967.py", input_content, expected_output)
