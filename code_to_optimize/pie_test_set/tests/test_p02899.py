from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02899_0():
    input_content = "3\n2 3 1"
    expected_output = "3 1 2"
    run_pie_test_case("../p02899.py", input_content, expected_output)


def test_problem_p02899_1():
    input_content = "5\n1 2 3 4 5"
    expected_output = "1 2 3 4 5"
    run_pie_test_case("../p02899.py", input_content, expected_output)


def test_problem_p02899_2():
    input_content = "3\n2 3 1"
    expected_output = "3 1 2"
    run_pie_test_case("../p02899.py", input_content, expected_output)


def test_problem_p02899_3():
    input_content = "8\n8 2 7 3 4 5 6 1"
    expected_output = "8 2 4 5 6 7 3 1"
    run_pie_test_case("../p02899.py", input_content, expected_output)
