from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03137_0():
    input_content = "2 5\n10 12 1 2 14"
    expected_output = "5"
    run_pie_test_case("../p03137.py", input_content, expected_output)


def test_problem_p03137_1():
    input_content = "3 7\n-10 -3 0 9 -100 2 17"
    expected_output = "19"
    run_pie_test_case("../p03137.py", input_content, expected_output)


def test_problem_p03137_2():
    input_content = "2 5\n10 12 1 2 14"
    expected_output = "5"
    run_pie_test_case("../p03137.py", input_content, expected_output)


def test_problem_p03137_3():
    input_content = "100 1\n-100000"
    expected_output = "0"
    run_pie_test_case("../p03137.py", input_content, expected_output)
