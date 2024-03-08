from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03287_0():
    input_content = "3 2\n4 1 5"
    expected_output = "3"
    run_pie_test_case("../p03287.py", input_content, expected_output)


def test_problem_p03287_1():
    input_content = "10 400000000\n1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000"
    expected_output = "25"
    run_pie_test_case("../p03287.py", input_content, expected_output)


def test_problem_p03287_2():
    input_content = "13 17\n29 7 5 7 9 51 7 13 8 55 42 9 81"
    expected_output = "6"
    run_pie_test_case("../p03287.py", input_content, expected_output)


def test_problem_p03287_3():
    input_content = "3 2\n4 1 5"
    expected_output = "3"
    run_pie_test_case("../p03287.py", input_content, expected_output)
