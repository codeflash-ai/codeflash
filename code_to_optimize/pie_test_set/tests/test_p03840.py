from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03840_0():
    input_content = "2 1 1 0 0 0 0"
    expected_output = "3"
    run_pie_test_case("../p03840.py", input_content, expected_output)


def test_problem_p03840_1():
    input_content = "2 1 1 0 0 0 0"
    expected_output = "3"
    run_pie_test_case("../p03840.py", input_content, expected_output)


def test_problem_p03840_2():
    input_content = "0 0 10 0 0 0 0"
    expected_output = "0"
    run_pie_test_case("../p03840.py", input_content, expected_output)
