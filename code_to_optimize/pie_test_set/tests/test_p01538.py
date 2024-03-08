from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01538_0():
    input_content = "3\n9\n99\n123"
    expected_output = "0\n2\n3"
    run_pie_test_case("../p01538.py", input_content, expected_output)


def test_problem_p01538_1():
    input_content = "2\n999999\n1000000"
    expected_output = "12\n1"
    run_pie_test_case("../p01538.py", input_content, expected_output)


def test_problem_p01538_2():
    input_content = "3\n9\n99\n123"
    expected_output = "0\n2\n3"
    run_pie_test_case("../p01538.py", input_content, expected_output)
