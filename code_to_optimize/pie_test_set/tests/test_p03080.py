from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03080_0():
    input_content = "4\nRRBR"
    expected_output = "Yes"
    run_pie_test_case("../p03080.py", input_content, expected_output)


def test_problem_p03080_1():
    input_content = "4\nRRBR"
    expected_output = "Yes"
    run_pie_test_case("../p03080.py", input_content, expected_output)


def test_problem_p03080_2():
    input_content = "4\nBRBR"
    expected_output = "No"
    run_pie_test_case("../p03080.py", input_content, expected_output)
