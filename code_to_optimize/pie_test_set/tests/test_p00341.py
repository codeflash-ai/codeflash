from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00341_0():
    input_content = "1 1 3 4 8 9 7 3 4 5 5 5"
    expected_output = "no"
    run_pie_test_case("../p00341.py", input_content, expected_output)


def test_problem_p00341_1():
    input_content = "1 1 2 2 3 1 2 3 3 3 1 2"
    expected_output = "yes"
    run_pie_test_case("../p00341.py", input_content, expected_output)


def test_problem_p00341_2():
    input_content = "1 1 3 4 8 9 7 3 4 5 5 5"
    expected_output = "no"
    run_pie_test_case("../p00341.py", input_content, expected_output)
