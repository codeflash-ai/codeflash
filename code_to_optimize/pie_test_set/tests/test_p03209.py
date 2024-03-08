from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03209_0():
    input_content = "2 7"
    expected_output = "4"
    run_pie_test_case("../p03209.py", input_content, expected_output)


def test_problem_p03209_1():
    input_content = "50 4321098765432109"
    expected_output = "2160549382716056"
    run_pie_test_case("../p03209.py", input_content, expected_output)


def test_problem_p03209_2():
    input_content = "1 1"
    expected_output = "0"
    run_pie_test_case("../p03209.py", input_content, expected_output)


def test_problem_p03209_3():
    input_content = "2 7"
    expected_output = "4"
    run_pie_test_case("../p03209.py", input_content, expected_output)
