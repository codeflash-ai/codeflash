from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03683_0():
    input_content = "2 2"
    expected_output = "8"
    run_pie_test_case("../p03683.py", input_content, expected_output)


def test_problem_p03683_1():
    input_content = "100000 100000"
    expected_output = "530123477"
    run_pie_test_case("../p03683.py", input_content, expected_output)


def test_problem_p03683_2():
    input_content = "3 2"
    expected_output = "12"
    run_pie_test_case("../p03683.py", input_content, expected_output)


def test_problem_p03683_3():
    input_content = "1 8"
    expected_output = "0"
    run_pie_test_case("../p03683.py", input_content, expected_output)


def test_problem_p03683_4():
    input_content = "2 2"
    expected_output = "8"
    run_pie_test_case("../p03683.py", input_content, expected_output)
