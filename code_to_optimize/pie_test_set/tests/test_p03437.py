from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03437_0():
    input_content = "8 6"
    expected_output = "16"
    run_pie_test_case("../p03437.py", input_content, expected_output)


def test_problem_p03437_1():
    input_content = "3 3"
    expected_output = "-1"
    run_pie_test_case("../p03437.py", input_content, expected_output)


def test_problem_p03437_2():
    input_content = "8 6"
    expected_output = "16"
    run_pie_test_case("../p03437.py", input_content, expected_output)
