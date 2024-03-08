from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03778_0():
    input_content = "3 2 6"
    expected_output = "1"
    run_pie_test_case("../p03778.py", input_content, expected_output)


def test_problem_p03778_1():
    input_content = "3 2 6"
    expected_output = "1"
    run_pie_test_case("../p03778.py", input_content, expected_output)


def test_problem_p03778_2():
    input_content = "3 1 3"
    expected_output = "0"
    run_pie_test_case("../p03778.py", input_content, expected_output)


def test_problem_p03778_3():
    input_content = "5 10 1"
    expected_output = "4"
    run_pie_test_case("../p03778.py", input_content, expected_output)
