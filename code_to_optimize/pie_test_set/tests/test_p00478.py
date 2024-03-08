from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00478_0():
    input_content = "ABCD\n3\nABCDXXXXXX\nYYYYABCDXX\nDCBAZZZZZZ"
    expected_output = "2"
    run_pie_test_case("../p00478.py", input_content, expected_output)
