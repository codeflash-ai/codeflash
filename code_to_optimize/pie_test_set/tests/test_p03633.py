from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03633_0():
    input_content = "2\n2\n3"
    expected_output = "6"
    run_pie_test_case("../p03633.py", input_content, expected_output)


def test_problem_p03633_1():
    input_content = "5\n2\n5\n10\n1000000000000000000\n1000000000000000000"
    expected_output = "1000000000000000000"
    run_pie_test_case("../p03633.py", input_content, expected_output)


def test_problem_p03633_2():
    input_content = "2\n2\n3"
    expected_output = "6"
    run_pie_test_case("../p03633.py", input_content, expected_output)
