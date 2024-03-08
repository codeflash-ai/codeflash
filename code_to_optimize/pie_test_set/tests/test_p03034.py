from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03034_0():
    input_content = "5\n0 2 5 1 0"
    expected_output = "3"
    run_pie_test_case("../p03034.py", input_content, expected_output)


def test_problem_p03034_1():
    input_content = "6\n0 10 -7 -4 -13 0"
    expected_output = "0"
    run_pie_test_case("../p03034.py", input_content, expected_output)


def test_problem_p03034_2():
    input_content = "5\n0 2 5 1 0"
    expected_output = "3"
    run_pie_test_case("../p03034.py", input_content, expected_output)


def test_problem_p03034_3():
    input_content = "11\n0 -4 0 -99 31 14 -15 -39 43 18 0"
    expected_output = "59"
    run_pie_test_case("../p03034.py", input_content, expected_output)
