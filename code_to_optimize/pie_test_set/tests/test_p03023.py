from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03023_0():
    input_content = "3"
    expected_output = "180"
    run_pie_test_case("../p03023.py", input_content, expected_output)


def test_problem_p03023_1():
    input_content = "3"
    expected_output = "180"
    run_pie_test_case("../p03023.py", input_content, expected_output)


def test_problem_p03023_2():
    input_content = "100"
    expected_output = "17640"
    run_pie_test_case("../p03023.py", input_content, expected_output)
