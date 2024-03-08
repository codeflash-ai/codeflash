from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02553_0():
    input_content = "1 2 1 1"
    expected_output = "2"
    run_pie_test_case("../p02553.py", input_content, expected_output)


def test_problem_p02553_1():
    input_content = "-1000000000 0 -1000000000 0"
    expected_output = "1000000000000000000"
    run_pie_test_case("../p02553.py", input_content, expected_output)


def test_problem_p02553_2():
    input_content = "1 2 1 1"
    expected_output = "2"
    run_pie_test_case("../p02553.py", input_content, expected_output)


def test_problem_p02553_3():
    input_content = "3 5 -4 -2"
    expected_output = "-6"
    run_pie_test_case("../p02553.py", input_content, expected_output)
