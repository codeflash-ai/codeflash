from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02276_0():
    input_content = "12\n13 19 9 5 12 8 7 4 21 2 6 11"
    expected_output = "9 5 8 7 4 2 6 [11] 21 13 19 12"
    run_pie_test_case("../p02276.py", input_content, expected_output)


def test_problem_p02276_1():
    input_content = "12\n13 19 9 5 12 8 7 4 21 2 6 11"
    expected_output = "9 5 8 7 4 2 6 [11] 21 13 19 12"
    run_pie_test_case("../p02276.py", input_content, expected_output)
