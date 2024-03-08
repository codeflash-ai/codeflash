from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03011_0():
    input_content = "1 3 4"
    expected_output = "4"
    run_pie_test_case("../p03011.py", input_content, expected_output)


def test_problem_p03011_1():
    input_content = "3 2 3"
    expected_output = "5"
    run_pie_test_case("../p03011.py", input_content, expected_output)


def test_problem_p03011_2():
    input_content = "1 3 4"
    expected_output = "4"
    run_pie_test_case("../p03011.py", input_content, expected_output)
