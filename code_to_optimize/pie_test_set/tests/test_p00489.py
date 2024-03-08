from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00489_0():
    input_content = "4\n1 2 0 1\n1 3 2 1\n1 4 2 2\n2 3 1 1\n2 4 3 0\n3 4 1 3"
    expected_output = "2\n1\n4\n2"
    run_pie_test_case("../p00489.py", input_content, expected_output)


def test_problem_p00489_1():
    input_content = "4\n1 2 0 1\n1 3 2 1\n1 4 2 2\n2 3 1 1\n2 4 3 0\n3 4 1 3"
    expected_output = "2\n1\n4\n2"
    run_pie_test_case("../p00489.py", input_content, expected_output)
