from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01300_0():
    input_content = "17819\n1111\n11011\n1234567891011121314151617181920\n0"
    expected_output = "1\n4\n4\n38"
    run_pie_test_case("../p01300.py", input_content, expected_output)


def test_problem_p01300_1():
    input_content = "17819\n1111\n11011\n1234567891011121314151617181920\n0"
    expected_output = "1\n4\n4\n38"
    run_pie_test_case("../p01300.py", input_content, expected_output)
