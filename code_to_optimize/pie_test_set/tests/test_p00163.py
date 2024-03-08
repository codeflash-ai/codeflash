from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00163_0():
    input_content = "2\n17 25\n4\n17 45\n4\n17 25\n7\n19 35\n0"
    expected_output = "250\n1300"
    run_pie_test_case("../p00163.py", input_content, expected_output)


def test_problem_p00163_1():
    input_content = "2\n17 25\n4\n17 45\n4\n17 25\n7\n19 35\n0"
    expected_output = "250\n1300"
    run_pie_test_case("../p00163.py", input_content, expected_output)
