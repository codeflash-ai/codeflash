from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00151_0():
    input_content = "5\n00011\n00101\n01000\n10101\n00010\n8\n11000001\n10110111\n01100111\n01111010\n11111111\n01011010\n10100010\n10000001\n2\n01\n00\n3\n000\n000\n000\n0"
    expected_output = "4\n8\n1\n0"
    run_pie_test_case("../p00151.py", input_content, expected_output)


def test_problem_p00151_1():
    input_content = "5\n00011\n00101\n01000\n10101\n00010\n8\n11000001\n10110111\n01100111\n01111010\n11111111\n01011010\n10100010\n10000001\n2\n01\n00\n3\n000\n000\n000\n0"
    expected_output = "4\n8\n1\n0"
    run_pie_test_case("../p00151.py", input_content, expected_output)
