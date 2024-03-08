from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00631_0():
    input_content = "5\n1 2 3 4 5\n4\n2 3 5 7\n0"
    expected_output = "1\n1"
    run_pie_test_case("../p00631.py", input_content, expected_output)


def test_problem_p00631_1():
    input_content = "5\n1 2 3 4 5\n4\n2 3 5 7\n0"
    expected_output = "1\n1"
    run_pie_test_case("../p00631.py", input_content, expected_output)
