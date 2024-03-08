from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03352_0():
    input_content = "10"
    expected_output = "9"
    run_pie_test_case("../p03352.py", input_content, expected_output)


def test_problem_p03352_1():
    input_content = "10"
    expected_output = "9"
    run_pie_test_case("../p03352.py", input_content, expected_output)


def test_problem_p03352_2():
    input_content = "1"
    expected_output = "1"
    run_pie_test_case("../p03352.py", input_content, expected_output)


def test_problem_p03352_3():
    input_content = "999"
    expected_output = "961"
    run_pie_test_case("../p03352.py", input_content, expected_output)
