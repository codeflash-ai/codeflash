from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00322_0():
    input_content = "7 6 -1 1 -1 9 2 3 4"
    expected_output = "1"
    run_pie_test_case("../p00322.py", input_content, expected_output)


def test_problem_p00322_1():
    input_content = "7 6 5 1 8 9 2 3 4"
    expected_output = "0"
    run_pie_test_case("../p00322.py", input_content, expected_output)


def test_problem_p00322_2():
    input_content = "-1 -1 -1 -1 -1 -1 8 4 6"
    expected_output = "12"
    run_pie_test_case("../p00322.py", input_content, expected_output)


def test_problem_p00322_3():
    input_content = "7 6 -1 1 -1 9 2 3 4"
    expected_output = "1"
    run_pie_test_case("../p00322.py", input_content, expected_output)


def test_problem_p00322_4():
    input_content = "-1 -1 -1 -1 -1 -1 -1 -1 -1"
    expected_output = "168"
    run_pie_test_case("../p00322.py", input_content, expected_output)
