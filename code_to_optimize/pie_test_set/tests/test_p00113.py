from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00113_0():
    input_content = "1 12\n10000 32768\n1 11100\n1 459550"
    expected_output = "083\n  ^\n30517578125\n00009\n  ^^^\n00000217604178\n  ^^^^^^^^^^^^"
    run_pie_test_case("../p00113.py", input_content, expected_output)
