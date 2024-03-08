from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03948_0():
    input_content = "3 2\n100 50 200"
    expected_output = "1"
    run_pie_test_case("../p03948.py", input_content, expected_output)


def test_problem_p03948_1():
    input_content = "5 8\n50 30 40 10 20"
    expected_output = "2"
    run_pie_test_case("../p03948.py", input_content, expected_output)


def test_problem_p03948_2():
    input_content = "10 100\n7 10 4 5 9 3 6 8 2 1"
    expected_output = "2"
    run_pie_test_case("../p03948.py", input_content, expected_output)


def test_problem_p03948_3():
    input_content = "3 2\n100 50 200"
    expected_output = "1"
    run_pie_test_case("../p03948.py", input_content, expected_output)
