from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03345_0():
    input_content = "1 2 3 1"
    expected_output = "1"
    run_pie_test_case("../p03345.py", input_content, expected_output)


def test_problem_p03345_1():
    input_content = "1 2 3 1"
    expected_output = "1"
    run_pie_test_case("../p03345.py", input_content, expected_output)


def test_problem_p03345_2():
    input_content = "2 3 2 0"
    expected_output = "-1"
    run_pie_test_case("../p03345.py", input_content, expected_output)


def test_problem_p03345_3():
    input_content = "1000000000 1000000000 1000000000 1000000000000000000"
    expected_output = "0"
    run_pie_test_case("../p03345.py", input_content, expected_output)
