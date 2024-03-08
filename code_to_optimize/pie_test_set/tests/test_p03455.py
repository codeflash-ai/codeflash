from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03455_0():
    input_content = "3 4"
    expected_output = "Even"
    run_pie_test_case("../p03455.py", input_content, expected_output)


def test_problem_p03455_1():
    input_content = "3 4"
    expected_output = "Even"
    run_pie_test_case("../p03455.py", input_content, expected_output)


def test_problem_p03455_2():
    input_content = "1 21"
    expected_output = "Odd"
    run_pie_test_case("../p03455.py", input_content, expected_output)
