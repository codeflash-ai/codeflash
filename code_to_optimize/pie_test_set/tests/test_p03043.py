from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03043_0():
    input_content = "3 10"
    expected_output = "0.145833333333"
    run_pie_test_case("../p03043.py", input_content, expected_output)


def test_problem_p03043_1():
    input_content = "3 10"
    expected_output = "0.145833333333"
    run_pie_test_case("../p03043.py", input_content, expected_output)


def test_problem_p03043_2():
    input_content = "100000 5"
    expected_output = "0.999973749998"
    run_pie_test_case("../p03043.py", input_content, expected_output)
