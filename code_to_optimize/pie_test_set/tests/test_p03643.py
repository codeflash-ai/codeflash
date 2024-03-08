from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03643_0():
    input_content = "100"
    expected_output = "ABC100"
    run_pie_test_case("../p03643.py", input_content, expected_output)


def test_problem_p03643_1():
    input_content = "999"
    expected_output = "ABC999"
    run_pie_test_case("../p03643.py", input_content, expected_output)


def test_problem_p03643_2():
    input_content = "425"
    expected_output = "ABC425"
    run_pie_test_case("../p03643.py", input_content, expected_output)


def test_problem_p03643_3():
    input_content = "100"
    expected_output = "ABC100"
    run_pie_test_case("../p03643.py", input_content, expected_output)
