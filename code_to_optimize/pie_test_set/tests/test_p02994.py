from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02994_0():
    input_content = "5 2"
    expected_output = "18"
    run_pie_test_case("../p02994.py", input_content, expected_output)


def test_problem_p02994_1():
    input_content = "5 2"
    expected_output = "18"
    run_pie_test_case("../p02994.py", input_content, expected_output)


def test_problem_p02994_2():
    input_content = "30 -50"
    expected_output = "-1044"
    run_pie_test_case("../p02994.py", input_content, expected_output)


def test_problem_p02994_3():
    input_content = "3 -1"
    expected_output = "0"
    run_pie_test_case("../p02994.py", input_content, expected_output)
