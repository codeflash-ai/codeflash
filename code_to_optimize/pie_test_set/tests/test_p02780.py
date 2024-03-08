from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02780_0():
    input_content = "5 3\n1 2 2 4 5"
    expected_output = "7.000000000000"
    run_pie_test_case("../p02780.py", input_content, expected_output)


def test_problem_p02780_1():
    input_content = "5 3\n1 2 2 4 5"
    expected_output = "7.000000000000"
    run_pie_test_case("../p02780.py", input_content, expected_output)


def test_problem_p02780_2():
    input_content = "10 4\n17 13 13 12 15 20 10 13 17 11"
    expected_output = "32.000000000000"
    run_pie_test_case("../p02780.py", input_content, expected_output)


def test_problem_p02780_3():
    input_content = "4 1\n6 6 6 6"
    expected_output = "3.500000000000"
    run_pie_test_case("../p02780.py", input_content, expected_output)
