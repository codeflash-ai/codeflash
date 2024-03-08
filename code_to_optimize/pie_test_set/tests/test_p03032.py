from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03032_0():
    input_content = "6 4\n-10 8 2 1 2 6"
    expected_output = "14"
    run_pie_test_case("../p03032.py", input_content, expected_output)


def test_problem_p03032_1():
    input_content = "6 4\n-10 8 2 1 2 6"
    expected_output = "14"
    run_pie_test_case("../p03032.py", input_content, expected_output)


def test_problem_p03032_2():
    input_content = "6 4\n-6 -100 50 -2 -5 -3"
    expected_output = "44"
    run_pie_test_case("../p03032.py", input_content, expected_output)


def test_problem_p03032_3():
    input_content = "6 3\n-6 -100 50 -2 -5 -3"
    expected_output = "0"
    run_pie_test_case("../p03032.py", input_content, expected_output)
