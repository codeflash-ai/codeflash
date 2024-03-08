from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03618_0():
    input_content = "aatt"
    expected_output = "5"
    run_pie_test_case("../p03618.py", input_content, expected_output)


def test_problem_p03618_1():
    input_content = "xxxxxxxxxx"
    expected_output = "1"
    run_pie_test_case("../p03618.py", input_content, expected_output)


def test_problem_p03618_2():
    input_content = "aatt"
    expected_output = "5"
    run_pie_test_case("../p03618.py", input_content, expected_output)


def test_problem_p03618_3():
    input_content = "abracadabra"
    expected_output = "44"
    run_pie_test_case("../p03618.py", input_content, expected_output)
