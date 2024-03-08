from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02950_0():
    input_content = "2\n1 0"
    expected_output = "1 1"
    run_pie_test_case("../p02950.py", input_content, expected_output)


def test_problem_p02950_1():
    input_content = "2\n1 0"
    expected_output = "1 1"
    run_pie_test_case("../p02950.py", input_content, expected_output)


def test_problem_p02950_2():
    input_content = "3\n0 0 0"
    expected_output = "0 0 0"
    run_pie_test_case("../p02950.py", input_content, expected_output)


def test_problem_p02950_3():
    input_content = "5\n0 1 0 1 0"
    expected_output = "0 2 0 1 3"
    run_pie_test_case("../p02950.py", input_content, expected_output)
