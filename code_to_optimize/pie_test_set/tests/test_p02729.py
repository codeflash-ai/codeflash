from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02729_0():
    input_content = "2 1"
    expected_output = "1"
    run_pie_test_case("../p02729.py", input_content, expected_output)


def test_problem_p02729_1():
    input_content = "13 3"
    expected_output = "81"
    run_pie_test_case("../p02729.py", input_content, expected_output)


def test_problem_p02729_2():
    input_content = "1 1"
    expected_output = "0"
    run_pie_test_case("../p02729.py", input_content, expected_output)


def test_problem_p02729_3():
    input_content = "4 3"
    expected_output = "9"
    run_pie_test_case("../p02729.py", input_content, expected_output)


def test_problem_p02729_4():
    input_content = "2 1"
    expected_output = "1"
    run_pie_test_case("../p02729.py", input_content, expected_output)


def test_problem_p02729_5():
    input_content = "0 3"
    expected_output = "3"
    run_pie_test_case("../p02729.py", input_content, expected_output)
