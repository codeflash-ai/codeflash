from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03125_0():
    input_content = "4 12"
    expected_output = "16"
    run_pie_test_case("../p03125.py", input_content, expected_output)


def test_problem_p03125_1():
    input_content = "4 12"
    expected_output = "16"
    run_pie_test_case("../p03125.py", input_content, expected_output)


def test_problem_p03125_2():
    input_content = "8 20"
    expected_output = "12"
    run_pie_test_case("../p03125.py", input_content, expected_output)


def test_problem_p03125_3():
    input_content = "1 1"
    expected_output = "2"
    run_pie_test_case("../p03125.py", input_content, expected_output)
