from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03288_0():
    input_content = "1199"
    expected_output = "ABC"
    run_pie_test_case("../p03288.py", input_content, expected_output)


def test_problem_p03288_1():
    input_content = "1199"
    expected_output = "ABC"
    run_pie_test_case("../p03288.py", input_content, expected_output)


def test_problem_p03288_2():
    input_content = "1200"
    expected_output = "ARC"
    run_pie_test_case("../p03288.py", input_content, expected_output)


def test_problem_p03288_3():
    input_content = "4208"
    expected_output = "AGC"
    run_pie_test_case("../p03288.py", input_content, expected_output)
