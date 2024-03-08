from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03548_0():
    input_content = "13 3 1"
    expected_output = "3"
    run_pie_test_case("../p03548.py", input_content, expected_output)


def test_problem_p03548_1():
    input_content = "100000 1 1"
    expected_output = "49999"
    run_pie_test_case("../p03548.py", input_content, expected_output)


def test_problem_p03548_2():
    input_content = "13 3 1"
    expected_output = "3"
    run_pie_test_case("../p03548.py", input_content, expected_output)


def test_problem_p03548_3():
    input_content = "64145 123 456"
    expected_output = "109"
    run_pie_test_case("../p03548.py", input_content, expected_output)


def test_problem_p03548_4():
    input_content = "12 3 1"
    expected_output = "2"
    run_pie_test_case("../p03548.py", input_content, expected_output)


def test_problem_p03548_5():
    input_content = "64146 123 456"
    expected_output = "110"
    run_pie_test_case("../p03548.py", input_content, expected_output)
