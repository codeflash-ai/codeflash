from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00098_0():
    input_content = "3\n1 -2 3\n-4 5 6\n7 8 -9"
    expected_output = "16"
    run_pie_test_case("../p00098.py", input_content, expected_output)


def test_problem_p00098_1():
    input_content = "3\n1 -2 3\n-4 5 6\n7 8 -9"
    expected_output = "16"
    run_pie_test_case("../p00098.py", input_content, expected_output)


def test_problem_p00098_2():
    input_content = "4\n1 3 -9 2\n2 7 -1 5\n-8 3 2 -1\n5 0 -3 1"
    expected_output = "15"
    run_pie_test_case("../p00098.py", input_content, expected_output)
