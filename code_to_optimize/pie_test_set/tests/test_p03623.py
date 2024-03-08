from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03623_0():
    input_content = "5 2 7"
    expected_output = "B"
    run_pie_test_case("../p03623.py", input_content, expected_output)


def test_problem_p03623_1():
    input_content = "1 999 1000"
    expected_output = "A"
    run_pie_test_case("../p03623.py", input_content, expected_output)


def test_problem_p03623_2():
    input_content = "5 2 7"
    expected_output = "B"
    run_pie_test_case("../p03623.py", input_content, expected_output)
