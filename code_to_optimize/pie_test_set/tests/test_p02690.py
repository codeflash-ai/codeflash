from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02690_0():
    input_content = "33"
    expected_output = "2 -1"
    run_pie_test_case("../p02690.py", input_content, expected_output)


def test_problem_p02690_1():
    input_content = "33"
    expected_output = "2 -1"
    run_pie_test_case("../p02690.py", input_content, expected_output)


def test_problem_p02690_2():
    input_content = "1"
    expected_output = "0 -1"
    run_pie_test_case("../p02690.py", input_content, expected_output)
