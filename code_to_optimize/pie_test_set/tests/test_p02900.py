from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02900_0():
    input_content = "12 18"
    expected_output = "3"
    run_pie_test_case("../p02900.py", input_content, expected_output)


def test_problem_p02900_1():
    input_content = "1 2019"
    expected_output = "1"
    run_pie_test_case("../p02900.py", input_content, expected_output)


def test_problem_p02900_2():
    input_content = "12 18"
    expected_output = "3"
    run_pie_test_case("../p02900.py", input_content, expected_output)


def test_problem_p02900_3():
    input_content = "420 660"
    expected_output = "4"
    run_pie_test_case("../p02900.py", input_content, expected_output)
