from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03696_0():
    input_content = "3\n())"
    expected_output = "(())"
    run_pie_test_case("../p03696.py", input_content, expected_output)


def test_problem_p03696_1():
    input_content = "8\n))))(((("
    expected_output = "(((())))(((())))"
    run_pie_test_case("../p03696.py", input_content, expected_output)


def test_problem_p03696_2():
    input_content = "3\n())"
    expected_output = "(())"
    run_pie_test_case("../p03696.py", input_content, expected_output)


def test_problem_p03696_3():
    input_content = "6\n)))())"
    expected_output = "(((()))())"
    run_pie_test_case("../p03696.py", input_content, expected_output)
