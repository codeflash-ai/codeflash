from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00107_0():
    input_content = "10 6 8\n5\n4\n8\n6\n2\n5\n0 0 0"
    expected_output = "NA\nOK\nOK\nNA\nNA"
    run_pie_test_case("../p00107.py", input_content, expected_output)


def test_problem_p00107_1():
    input_content = "10 6 8\n5\n4\n8\n6\n2\n5\n0 0 0"
    expected_output = "NA\nOK\nOK\nNA\nNA"
    run_pie_test_case("../p00107.py", input_content, expected_output)
