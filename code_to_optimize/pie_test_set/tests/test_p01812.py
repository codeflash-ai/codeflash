from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01812_0():
    input_content = "4 2 2\n1 2\n2 4\n3 1\n4 2\n1 3"
    expected_output = "2"
    run_pie_test_case("../p01812.py", input_content, expected_output)


def test_problem_p01812_1():
    input_content = "4 2 2\n1 2\n2 4\n3 1\n4 2\n1 3"
    expected_output = "2"
    run_pie_test_case("../p01812.py", input_content, expected_output)
