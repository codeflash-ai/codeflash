from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02832_0():
    input_content = "3\n2 1 2"
    expected_output = "1"
    run_pie_test_case("../p02832.py", input_content, expected_output)


def test_problem_p02832_1():
    input_content = "3\n2 2 2"
    expected_output = "-1"
    run_pie_test_case("../p02832.py", input_content, expected_output)


def test_problem_p02832_2():
    input_content = "10\n3 1 4 1 5 9 2 6 5 3"
    expected_output = "7"
    run_pie_test_case("../p02832.py", input_content, expected_output)


def test_problem_p02832_3():
    input_content = "1\n1"
    expected_output = "0"
    run_pie_test_case("../p02832.py", input_content, expected_output)


def test_problem_p02832_4():
    input_content = "3\n2 1 2"
    expected_output = "1"
    run_pie_test_case("../p02832.py", input_content, expected_output)
