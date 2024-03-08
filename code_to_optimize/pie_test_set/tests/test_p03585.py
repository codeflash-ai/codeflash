from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03585_0():
    input_content = "3\n1 1 1\n2 -1 2\n-1 2 2"
    expected_output = "1.000000000000000 1.000000000000000"
    run_pie_test_case("../p03585.py", input_content, expected_output)


def test_problem_p03585_1():
    input_content = "4\n1 1 2\n1 -1 0\n3 -1 -2\n1 -3 4"
    expected_output = "-1.000000000000000 -1.000000000000000"
    run_pie_test_case("../p03585.py", input_content, expected_output)


def test_problem_p03585_2():
    input_content = "3\n1 1 1\n2 -1 2\n-1 2 2"
    expected_output = "1.000000000000000 1.000000000000000"
    run_pie_test_case("../p03585.py", input_content, expected_output)


def test_problem_p03585_3():
    input_content = "7\n1 7 8\n-2 4 9\n3 -8 -5\n9 2 -14\n6 7 5\n-8 -9 3\n3 8 10"
    expected_output = "-1.722222222222222 1.325000000000000"
    run_pie_test_case("../p03585.py", input_content, expected_output)
