from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00168_0():
    input_content = "1\n10\n20\n25\n0"
    expected_output = "1\n1\n34\n701"
    run_pie_test_case("../p00168.py", input_content, expected_output)


def test_problem_p00168_1():
    input_content = "1\n10\n20\n25\n0"
    expected_output = "1\n1\n34\n701"
    run_pie_test_case("../p00168.py", input_content, expected_output)
