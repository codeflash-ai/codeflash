from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00511_0():
    input_content = "1\n+\n1\n="
    expected_output = "2"
    run_pie_test_case("../p00511.py", input_content, expected_output)
