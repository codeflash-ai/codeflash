from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02658_0():
    input_content = "2\n1000000000 1000000000"
    expected_output = "1000000000000000000"
    run_pie_test_case("../p02658.py", input_content, expected_output)


def test_problem_p02658_1():
    input_content = "3\n101 9901 999999000001"
    expected_output = "-1"
    run_pie_test_case("../p02658.py", input_content, expected_output)


def test_problem_p02658_2():
    input_content = "2\n1000000000 1000000000"
    expected_output = "1000000000000000000"
    run_pie_test_case("../p02658.py", input_content, expected_output)


def test_problem_p02658_3():
    input_content = "31\n4 1 5 9 2 6 5 3 5 8 9 7 9 3 2 3 8 4 6 2 6 4 3 3 8 3 2 7 9 5 0"
    expected_output = "0"
    run_pie_test_case("../p02658.py", input_content, expected_output)
