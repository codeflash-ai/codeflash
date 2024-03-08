from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03503_0():
    input_content = "1\n1 1 0 1 0 0 0 1 0 1\n3 4 5 6 7 8 9 -2 -3 4 -2"
    expected_output = "8"
    run_pie_test_case("../p03503.py", input_content, expected_output)


def test_problem_p03503_1():
    input_content = "1\n1 1 0 1 0 0 0 1 0 1\n3 4 5 6 7 8 9 -2 -3 4 -2"
    expected_output = "8"
    run_pie_test_case("../p03503.py", input_content, expected_output)


def test_problem_p03503_2():
    input_content = "3\n1 1 1 1 1 1 0 0 1 1\n0 1 0 1 1 1 1 0 1 0\n1 0 1 1 0 1 0 1 0 1\n-8 6 -2 -8 -8 4 8 7 -6 2 2\n-9 2 0 1 7 -5 0 -2 -6 5 5\n6 -6 7 -9 6 -5 8 0 -9 -7 -7"
    expected_output = "23"
    run_pie_test_case("../p03503.py", input_content, expected_output)


def test_problem_p03503_3():
    input_content = "2\n1 1 1 1 1 0 0 0 0 0\n0 0 0 0 0 1 1 1 1 1\n0 -2 -2 -2 -2 -2 -1 -1 -1 -1 -1\n0 -2 -2 -2 -2 -2 -1 -1 -1 -1 -1"
    expected_output = "-2"
    run_pie_test_case("../p03503.py", input_content, expected_output)
