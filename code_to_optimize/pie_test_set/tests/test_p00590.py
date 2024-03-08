from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00590_0():
    input_content = "1\n4\n7\n51"
    expected_output = "0\n2\n2\n6"
    run_pie_test_case("../p00590.py", input_content, expected_output)


def test_problem_p00590_1():
    input_content = "1\n4\n7\n51"
    expected_output = "0\n2\n2\n6"
    run_pie_test_case("../p00590.py", input_content, expected_output)
