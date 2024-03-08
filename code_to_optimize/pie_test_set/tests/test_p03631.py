from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03631_0():
    input_content = "575"
    expected_output = "Yes"
    run_pie_test_case("../p03631.py", input_content, expected_output)


def test_problem_p03631_1():
    input_content = "812"
    expected_output = "No"
    run_pie_test_case("../p03631.py", input_content, expected_output)


def test_problem_p03631_2():
    input_content = "575"
    expected_output = "Yes"
    run_pie_test_case("../p03631.py", input_content, expected_output)


def test_problem_p03631_3():
    input_content = "123"
    expected_output = "No"
    run_pie_test_case("../p03631.py", input_content, expected_output)
