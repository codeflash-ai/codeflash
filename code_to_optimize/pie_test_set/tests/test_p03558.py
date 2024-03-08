from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03558_0():
    input_content = "6"
    expected_output = "3"
    run_pie_test_case("../p03558.py", input_content, expected_output)


def test_problem_p03558_1():
    input_content = "41"
    expected_output = "5"
    run_pie_test_case("../p03558.py", input_content, expected_output)


def test_problem_p03558_2():
    input_content = "6"
    expected_output = "3"
    run_pie_test_case("../p03558.py", input_content, expected_output)


def test_problem_p03558_3():
    input_content = "79992"
    expected_output = "36"
    run_pie_test_case("../p03558.py", input_content, expected_output)
