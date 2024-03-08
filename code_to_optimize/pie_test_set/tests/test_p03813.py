from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03813_0():
    input_content = "1000"
    expected_output = "ABC"
    run_pie_test_case("../p03813.py", input_content, expected_output)


def test_problem_p03813_1():
    input_content = "1000"
    expected_output = "ABC"
    run_pie_test_case("../p03813.py", input_content, expected_output)


def test_problem_p03813_2():
    input_content = "2000"
    expected_output = "ARC"
    run_pie_test_case("../p03813.py", input_content, expected_output)
