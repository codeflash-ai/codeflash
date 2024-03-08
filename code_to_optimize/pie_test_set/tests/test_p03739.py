from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03739_0():
    input_content = "4\n1 -3 1 0"
    expected_output = "4"
    run_pie_test_case("../p03739.py", input_content, expected_output)


def test_problem_p03739_1():
    input_content = "5\n3 -6 4 -5 7"
    expected_output = "0"
    run_pie_test_case("../p03739.py", input_content, expected_output)


def test_problem_p03739_2():
    input_content = "4\n1 -3 1 0"
    expected_output = "4"
    run_pie_test_case("../p03739.py", input_content, expected_output)


def test_problem_p03739_3():
    input_content = "6\n-1 4 3 2 -5 4"
    expected_output = "8"
    run_pie_test_case("../p03739.py", input_content, expected_output)
