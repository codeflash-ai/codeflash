from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00254_0():
    input_content = "6174\n2012\n3333\n0000"
    expected_output = "0\n3\nNA"
    run_pie_test_case("../p00254.py", input_content, expected_output)


def test_problem_p00254_1():
    input_content = "6174\n2012\n3333\n0000"
    expected_output = "0\n3\nNA"
    run_pie_test_case("../p00254.py", input_content, expected_output)
