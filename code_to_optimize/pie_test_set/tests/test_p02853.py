from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02853_0():
    input_content = "1 1"
    expected_output = "1000000"
    run_pie_test_case("../p02853.py", input_content, expected_output)


def test_problem_p02853_1():
    input_content = "3 101"
    expected_output = "100000"
    run_pie_test_case("../p02853.py", input_content, expected_output)


def test_problem_p02853_2():
    input_content = "1 1"
    expected_output = "1000000"
    run_pie_test_case("../p02853.py", input_content, expected_output)


def test_problem_p02853_3():
    input_content = "4 4"
    expected_output = "0"
    run_pie_test_case("../p02853.py", input_content, expected_output)
