from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03007_0():
    input_content = "3\n1 -1 2"
    expected_output = "4\n-1 1\n2 -2"
    run_pie_test_case("../p03007.py", input_content, expected_output)


def test_problem_p03007_1():
    input_content = "3\n1 1 1"
    expected_output = "1\n1 1\n1 0"
    run_pie_test_case("../p03007.py", input_content, expected_output)


def test_problem_p03007_2():
    input_content = "3\n1 -1 2"
    expected_output = "4\n-1 1\n2 -2"
    run_pie_test_case("../p03007.py", input_content, expected_output)
