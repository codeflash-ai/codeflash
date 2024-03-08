from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03197_0():
    input_content = "2\n1\n2"
    expected_output = "first"
    run_pie_test_case("../p03197.py", input_content, expected_output)


def test_problem_p03197_1():
    input_content = "3\n100000\n30000\n20000"
    expected_output = "second"
    run_pie_test_case("../p03197.py", input_content, expected_output)


def test_problem_p03197_2():
    input_content = "2\n1\n2"
    expected_output = "first"
    run_pie_test_case("../p03197.py", input_content, expected_output)
