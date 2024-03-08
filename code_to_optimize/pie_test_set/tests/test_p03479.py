from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03479_0():
    input_content = "3 20"
    expected_output = "3"
    run_pie_test_case("../p03479.py", input_content, expected_output)


def test_problem_p03479_1():
    input_content = "3 20"
    expected_output = "3"
    run_pie_test_case("../p03479.py", input_content, expected_output)


def test_problem_p03479_2():
    input_content = "314159265 358979323846264338"
    expected_output = "31"
    run_pie_test_case("../p03479.py", input_content, expected_output)


def test_problem_p03479_3():
    input_content = "25 100"
    expected_output = "3"
    run_pie_test_case("../p03479.py", input_content, expected_output)
