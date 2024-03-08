from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02864_0():
    input_content = "4 1\n2 3 4 1"
    expected_output = "3"
    run_pie_test_case("../p02864.py", input_content, expected_output)


def test_problem_p02864_1():
    input_content = "10 0\n1 1000000000 1 1000000000 1 1000000000 1 1000000000 1 1000000000"
    expected_output = "4999999996"
    run_pie_test_case("../p02864.py", input_content, expected_output)


def test_problem_p02864_2():
    input_content = "6 2\n8 6 9 1 2 1"
    expected_output = "7"
    run_pie_test_case("../p02864.py", input_content, expected_output)


def test_problem_p02864_3():
    input_content = "4 1\n2 3 4 1"
    expected_output = "3"
    run_pie_test_case("../p02864.py", input_content, expected_output)
