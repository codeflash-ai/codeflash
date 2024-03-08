from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03284_0():
    input_content = "7 3"
    expected_output = "1"
    run_pie_test_case("../p03284.py", input_content, expected_output)


def test_problem_p03284_1():
    input_content = "100 10"
    expected_output = "0"
    run_pie_test_case("../p03284.py", input_content, expected_output)


def test_problem_p03284_2():
    input_content = "1 1"
    expected_output = "0"
    run_pie_test_case("../p03284.py", input_content, expected_output)


def test_problem_p03284_3():
    input_content = "7 3"
    expected_output = "1"
    run_pie_test_case("../p03284.py", input_content, expected_output)
