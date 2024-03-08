from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03325_0():
    input_content = "3\n5 2 4"
    expected_output = "3"
    run_pie_test_case("../p03325.py", input_content, expected_output)


def test_problem_p03325_1():
    input_content = "10\n2184 2126 1721 1800 1024 2528 3360 1945 1280 1776"
    expected_output = "39"
    run_pie_test_case("../p03325.py", input_content, expected_output)


def test_problem_p03325_2():
    input_content = "4\n631 577 243 199"
    expected_output = "0"
    run_pie_test_case("../p03325.py", input_content, expected_output)


def test_problem_p03325_3():
    input_content = "3\n5 2 4"
    expected_output = "3"
    run_pie_test_case("../p03325.py", input_content, expected_output)
