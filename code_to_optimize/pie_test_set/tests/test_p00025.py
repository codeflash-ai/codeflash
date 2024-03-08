from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00025_0():
    input_content = "9 1 8 2\n4 1 5 9\n4 6 8 2\n4 6 3 2"
    expected_output = "1 1\n3 0"
    run_pie_test_case("../p00025.py", input_content, expected_output)


def test_problem_p00025_1():
    input_content = "9 1 8 2\n4 1 5 9\n4 6 8 2\n4 6 3 2"
    expected_output = "1 1\n3 0"
    run_pie_test_case("../p00025.py", input_content, expected_output)
