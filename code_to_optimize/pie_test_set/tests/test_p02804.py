from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02804_0():
    input_content = "4 2\n1 1 3 4"
    expected_output = "11"
    run_pie_test_case("../p02804.py", input_content, expected_output)


def test_problem_p02804_1():
    input_content = "3 1\n1 1 1"
    expected_output = "0"
    run_pie_test_case("../p02804.py", input_content, expected_output)


def test_problem_p02804_2():
    input_content = "4 2\n1 1 3 4"
    expected_output = "11"
    run_pie_test_case("../p02804.py", input_content, expected_output)


def test_problem_p02804_3():
    input_content = "10 6\n1000000000 1000000000 1000000000 1000000000 1000000000 0 0 0 0 0"
    expected_output = "999998537"
    run_pie_test_case("../p02804.py", input_content, expected_output)


def test_problem_p02804_4():
    input_content = "6 3\n10 10 10 -10 -10 -10"
    expected_output = "360"
    run_pie_test_case("../p02804.py", input_content, expected_output)
