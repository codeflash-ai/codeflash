from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00162_0():
    input_content = "3 8\n1 27\n1 86\n0"
    expected_output = "5\n17\n31"
    run_pie_test_case("../p00162.py", input_content, expected_output)


def test_problem_p00162_1():
    input_content = "3 8\n1 27\n1 86\n0"
    expected_output = "5\n17\n31"
    run_pie_test_case("../p00162.py", input_content, expected_output)
