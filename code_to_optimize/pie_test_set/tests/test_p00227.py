from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00227_0():
    input_content = "4 2\n50 40 100 80\n7 3\n400 300 100 700 200 600 500\n0 0"
    expected_output = "150\n2100"
    run_pie_test_case("../p00227.py", input_content, expected_output)


def test_problem_p00227_1():
    input_content = "4 2\n50 40 100 80\n7 3\n400 300 100 700 200 600 500\n0 0"
    expected_output = "150\n2100"
    run_pie_test_case("../p00227.py", input_content, expected_output)
