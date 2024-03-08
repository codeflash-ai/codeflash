from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02282_0():
    input_content = "5\n1 2 3 4 5\n3 2 4 1 5"
    expected_output = "3 4 2 5 1"
    run_pie_test_case("../p02282.py", input_content, expected_output)


def test_problem_p02282_1():
    input_content = "5\n1 2 3 4 5\n3 2 4 1 5"
    expected_output = "3 4 2 5 1"
    run_pie_test_case("../p02282.py", input_content, expected_output)


def test_problem_p02282_2():
    input_content = "4\n1 2 3 4\n1 2 3 4"
    expected_output = "4 3 2 1"
    run_pie_test_case("../p02282.py", input_content, expected_output)
