from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03044_0():
    input_content = "3\n1 2 2\n2 3 1"
    expected_output = "0\n0\n1"
    run_pie_test_case("../p03044.py", input_content, expected_output)


def test_problem_p03044_1():
    input_content = "5\n2 5 2\n2 3 10\n1 3 8\n3 4 2"
    expected_output = "1\n0\n1\n0\n1"
    run_pie_test_case("../p03044.py", input_content, expected_output)


def test_problem_p03044_2():
    input_content = "3\n1 2 2\n2 3 1"
    expected_output = "0\n0\n1"
    run_pie_test_case("../p03044.py", input_content, expected_output)
