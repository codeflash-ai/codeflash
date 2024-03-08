from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00013_0():
    input_content = "1\n6\n0\n8\n10\n0\n0\n0"
    expected_output = "6\n10\n8\n1"
    run_pie_test_case("../p00013.py", input_content, expected_output)


def test_problem_p00013_1():
    input_content = "1\n6\n0\n8\n10\n0\n0\n0"
    expected_output = "6\n10\n8\n1"
    run_pie_test_case("../p00013.py", input_content, expected_output)
