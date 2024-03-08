from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03427_0():
    input_content = "100"
    expected_output = "18"
    run_pie_test_case("../p03427.py", input_content, expected_output)


def test_problem_p03427_1():
    input_content = "3141592653589793"
    expected_output = "137"
    run_pie_test_case("../p03427.py", input_content, expected_output)


def test_problem_p03427_2():
    input_content = "100"
    expected_output = "18"
    run_pie_test_case("../p03427.py", input_content, expected_output)


def test_problem_p03427_3():
    input_content = "9995"
    expected_output = "35"
    run_pie_test_case("../p03427.py", input_content, expected_output)
