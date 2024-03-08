from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00340_0():
    input_content = "1 1 3 4"
    expected_output = "no"
    run_pie_test_case("../p00340.py", input_content, expected_output)


def test_problem_p00340_1():
    input_content = "1 1 2 2"
    expected_output = "yes"
    run_pie_test_case("../p00340.py", input_content, expected_output)


def test_problem_p00340_2():
    input_content = "4 4 4 10"
    expected_output = "no"
    run_pie_test_case("../p00340.py", input_content, expected_output)


def test_problem_p00340_3():
    input_content = "2 1 1 2"
    expected_output = "yes"
    run_pie_test_case("../p00340.py", input_content, expected_output)


def test_problem_p00340_4():
    input_content = "1 1 3 4"
    expected_output = "no"
    run_pie_test_case("../p00340.py", input_content, expected_output)
