from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03711_0():
    input_content = "1 3"
    expected_output = "Yes"
    run_pie_test_case("../p03711.py", input_content, expected_output)


def test_problem_p03711_1():
    input_content = "2 4"
    expected_output = "No"
    run_pie_test_case("../p03711.py", input_content, expected_output)


def test_problem_p03711_2():
    input_content = "1 3"
    expected_output = "Yes"
    run_pie_test_case("../p03711.py", input_content, expected_output)
