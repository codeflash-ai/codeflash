from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03085_0():
    input_content = "A"
    expected_output = "T"
    run_pie_test_case("../p03085.py", input_content, expected_output)


def test_problem_p03085_1():
    input_content = "G"
    expected_output = "C"
    run_pie_test_case("../p03085.py", input_content, expected_output)


def test_problem_p03085_2():
    input_content = "A"
    expected_output = "T"
    run_pie_test_case("../p03085.py", input_content, expected_output)
