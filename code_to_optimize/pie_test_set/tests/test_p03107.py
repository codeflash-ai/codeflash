from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03107_0():
    input_content = "0011"
    expected_output = "4"
    run_pie_test_case("../p03107.py", input_content, expected_output)


def test_problem_p03107_1():
    input_content = "0"
    expected_output = "0"
    run_pie_test_case("../p03107.py", input_content, expected_output)


def test_problem_p03107_2():
    input_content = "0011"
    expected_output = "4"
    run_pie_test_case("../p03107.py", input_content, expected_output)


def test_problem_p03107_3():
    input_content = "11011010001011"
    expected_output = "12"
    run_pie_test_case("../p03107.py", input_content, expected_output)
