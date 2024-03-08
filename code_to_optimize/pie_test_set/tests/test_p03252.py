from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03252_0():
    input_content = "azzel\napple"
    expected_output = "Yes"
    run_pie_test_case("../p03252.py", input_content, expected_output)


def test_problem_p03252_1():
    input_content = "azzel\napple"
    expected_output = "Yes"
    run_pie_test_case("../p03252.py", input_content, expected_output)


def test_problem_p03252_2():
    input_content = "abcdefghijklmnopqrstuvwxyz\nibyhqfrekavclxjstdwgpzmonu"
    expected_output = "Yes"
    run_pie_test_case("../p03252.py", input_content, expected_output)


def test_problem_p03252_3():
    input_content = "chokudai\nredcoder"
    expected_output = "No"
    run_pie_test_case("../p03252.py", input_content, expected_output)
