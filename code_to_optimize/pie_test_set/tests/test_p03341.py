from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03341_0():
    input_content = "5\nWEEWW"
    expected_output = "1"
    run_pie_test_case("../p03341.py", input_content, expected_output)


def test_problem_p03341_1():
    input_content = "12\nWEWEWEEEWWWE"
    expected_output = "4"
    run_pie_test_case("../p03341.py", input_content, expected_output)


def test_problem_p03341_2():
    input_content = "5\nWEEWW"
    expected_output = "1"
    run_pie_test_case("../p03341.py", input_content, expected_output)


def test_problem_p03341_3():
    input_content = "8\nWWWWWEEE"
    expected_output = "3"
    run_pie_test_case("../p03341.py", input_content, expected_output)
