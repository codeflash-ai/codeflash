from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00855_0():
    input_content = "10\n11\n27\n2\n492170\n0"
    expected_output = "4\n0\n6\n0\n114"
    run_pie_test_case("../p00855.py", input_content, expected_output)


def test_problem_p00855_1():
    input_content = "10\n11\n27\n2\n492170\n0"
    expected_output = "4\n0\n6\n0\n114"
    run_pie_test_case("../p00855.py", input_content, expected_output)
