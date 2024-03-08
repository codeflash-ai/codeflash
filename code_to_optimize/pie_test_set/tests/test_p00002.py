from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00002_0():
    input_content = "5 7\n1 99\n1000 999"
    expected_output = "2\n3\n4"
    run_pie_test_case("../p00002.py", input_content, expected_output)


def test_problem_p00002_1():
    input_content = "5 7\n1 99\n1000 999"
    expected_output = "2\n3\n4"
    run_pie_test_case("../p00002.py", input_content, expected_output)
