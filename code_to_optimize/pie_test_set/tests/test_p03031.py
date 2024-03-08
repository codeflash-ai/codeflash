from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03031_0():
    input_content = "2 2\n2 1 2\n1 2\n0 1"
    expected_output = "1"
    run_pie_test_case("../p03031.py", input_content, expected_output)


def test_problem_p03031_1():
    input_content = "2 2\n2 1 2\n1 2\n0 1"
    expected_output = "1"
    run_pie_test_case("../p03031.py", input_content, expected_output)


def test_problem_p03031_2():
    input_content = "5 2\n3 1 2 5\n2 2 3\n1 0"
    expected_output = "8"
    run_pie_test_case("../p03031.py", input_content, expected_output)


def test_problem_p03031_3():
    input_content = "2 3\n2 1 2\n1 1\n1 2\n0 0 1"
    expected_output = "0"
    run_pie_test_case("../p03031.py", input_content, expected_output)
