from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03837_0():
    input_content = "3 3\n1 2 1\n1 3 1\n2 3 3"
    expected_output = "1"
    run_pie_test_case("../p03837.py", input_content, expected_output)


def test_problem_p03837_1():
    input_content = "3 3\n1 2 1\n1 3 1\n2 3 3"
    expected_output = "1"
    run_pie_test_case("../p03837.py", input_content, expected_output)


def test_problem_p03837_2():
    input_content = "3 2\n1 2 1\n2 3 1"
    expected_output = "0"
    run_pie_test_case("../p03837.py", input_content, expected_output)
