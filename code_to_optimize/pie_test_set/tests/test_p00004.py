from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00004_0():
    input_content = "1 2 3 4 5 6\n2 -1 -2 -1 -1 -5"
    expected_output = "-1.000 2.000\n1.000 4.000"
    run_pie_test_case("../p00004.py", input_content, expected_output)


def test_problem_p00004_1():
    input_content = "1 2 3 4 5 6\n2 -1 -2 -1 -1 -5"
    expected_output = "-1.000 2.000\n1.000 4.000"
    run_pie_test_case("../p00004.py", input_content, expected_output)


def test_problem_p00004_2():
    input_content = "2 -1 -3 1 -1 -3\n2 -1 -3 -9 9 27"
    expected_output = "0.000 3.000\n0.000 3.000"
    run_pie_test_case("../p00004.py", input_content, expected_output)
