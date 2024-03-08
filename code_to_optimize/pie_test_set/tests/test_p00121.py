from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00121_0():
    input_content = "0 1 2 3 4 5 6 7\n1 0 2 3 4 5 6 7\n7 6 5 4 3 2 1 0"
    expected_output = "0\n1\n28"
    run_pie_test_case("../p00121.py", input_content, expected_output)


def test_problem_p00121_1():
    input_content = "0 1 2 3 4 5 6 7\n1 0 2 3 4 5 6 7\n7 6 5 4 3 2 1 0"
    expected_output = "0\n1\n28"
    run_pie_test_case("../p00121.py", input_content, expected_output)
