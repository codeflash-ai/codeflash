from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03146_0():
    input_content = "8"
    expected_output = "5"
    run_pie_test_case("../p03146.py", input_content, expected_output)


def test_problem_p03146_1():
    input_content = "54"
    expected_output = "114"
    run_pie_test_case("../p03146.py", input_content, expected_output)


def test_problem_p03146_2():
    input_content = "8"
    expected_output = "5"
    run_pie_test_case("../p03146.py", input_content, expected_output)


def test_problem_p03146_3():
    input_content = "7"
    expected_output = "18"
    run_pie_test_case("../p03146.py", input_content, expected_output)
