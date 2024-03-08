from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03253_0():
    input_content = "2 6"
    expected_output = "4"
    run_pie_test_case("../p03253.py", input_content, expected_output)


def test_problem_p03253_1():
    input_content = "3 12"
    expected_output = "18"
    run_pie_test_case("../p03253.py", input_content, expected_output)


def test_problem_p03253_2():
    input_content = "2 6"
    expected_output = "4"
    run_pie_test_case("../p03253.py", input_content, expected_output)


def test_problem_p03253_3():
    input_content = "100000 1000000000"
    expected_output = "957870001"
    run_pie_test_case("../p03253.py", input_content, expected_output)
