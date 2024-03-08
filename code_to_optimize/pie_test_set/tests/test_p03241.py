from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03241_0():
    input_content = "3 14"
    expected_output = "2"
    run_pie_test_case("../p03241.py", input_content, expected_output)


def test_problem_p03241_1():
    input_content = "10 123"
    expected_output = "3"
    run_pie_test_case("../p03241.py", input_content, expected_output)


def test_problem_p03241_2():
    input_content = "3 14"
    expected_output = "2"
    run_pie_test_case("../p03241.py", input_content, expected_output)


def test_problem_p03241_3():
    input_content = "100000 1000000000"
    expected_output = "10000"
    run_pie_test_case("../p03241.py", input_content, expected_output)
