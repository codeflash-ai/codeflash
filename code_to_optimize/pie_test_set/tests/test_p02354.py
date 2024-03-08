from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02354_0():
    input_content = "6 4\n1 2 1 2 3 2"
    expected_output = "2"
    run_pie_test_case("../p02354.py", input_content, expected_output)


def test_problem_p02354_1():
    input_content = "6 4\n1 2 1 2 3 2"
    expected_output = "2"
    run_pie_test_case("../p02354.py", input_content, expected_output)


def test_problem_p02354_2():
    input_content = "6 6\n1 2 1 2 3 2"
    expected_output = "3"
    run_pie_test_case("../p02354.py", input_content, expected_output)


def test_problem_p02354_3():
    input_content = "3 7\n1 2 3"
    expected_output = "0"
    run_pie_test_case("../p02354.py", input_content, expected_output)
