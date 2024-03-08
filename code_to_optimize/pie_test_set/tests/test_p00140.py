from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00140_0():
    input_content = "2\n2 4\n4 2"
    expected_output = "2 3 4\n4 3 2"
    run_pie_test_case("../p00140.py", input_content, expected_output)


def test_problem_p00140_1():
    input_content = "2\n2 4\n4 2"
    expected_output = "2 3 4\n4 3 2"
    run_pie_test_case("../p00140.py", input_content, expected_output)
