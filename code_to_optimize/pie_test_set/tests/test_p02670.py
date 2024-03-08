from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02670_0():
    input_content = "3\n1 3 7 9 5 4 8 6 2"
    expected_output = "1"
    run_pie_test_case("../p02670.py", input_content, expected_output)


def test_problem_p02670_1():
    input_content = "4\n6 7 1 4 13 16 10 9 5 11 12 14 15 2 3 8"
    expected_output = "3"
    run_pie_test_case("../p02670.py", input_content, expected_output)


def test_problem_p02670_2():
    input_content = "6\n11 21 35 22 7 36 27 34 8 20 15 13 16 1 24 3 2 17 26 9 18 32 31 23 19 14 4 25 10 29 28 33 12 6 5 30"
    expected_output = "11"
    run_pie_test_case("../p02670.py", input_content, expected_output)


def test_problem_p02670_3():
    input_content = "3\n1 3 7 9 5 4 8 6 2"
    expected_output = "1"
    run_pie_test_case("../p02670.py", input_content, expected_output)
