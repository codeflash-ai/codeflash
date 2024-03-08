from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03206_0():
    input_content = "25"
    expected_output = "Christmas"
    run_pie_test_case("../p03206.py", input_content, expected_output)


def test_problem_p03206_1():
    input_content = "25"
    expected_output = "Christmas"
    run_pie_test_case("../p03206.py", input_content, expected_output)


def test_problem_p03206_2():
    input_content = "22"
    expected_output = "Christmas Eve Eve Eve"
    run_pie_test_case("../p03206.py", input_content, expected_output)
