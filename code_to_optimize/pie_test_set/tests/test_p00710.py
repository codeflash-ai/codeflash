from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00710_0():
    input_content = "5 2\n3 1\n3 1\n10 3\n1 10\n10 1\n8 3\n0 0"
    expected_output = "4\n4"
    run_pie_test_case("../p00710.py", input_content, expected_output)


def test_problem_p00710_1():
    input_content = "5 2\n3 1\n3 1\n10 3\n1 10\n10 1\n8 3\n0 0"
    expected_output = "4\n4"
    run_pie_test_case("../p00710.py", input_content, expected_output)
