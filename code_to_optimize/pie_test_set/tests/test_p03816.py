from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03816_0():
    input_content = "5\n1 2 1 3 7"
    expected_output = "3"
    run_pie_test_case("../p03816.py", input_content, expected_output)


def test_problem_p03816_1():
    input_content = "5\n1 2 1 3 7"
    expected_output = "3"
    run_pie_test_case("../p03816.py", input_content, expected_output)


def test_problem_p03816_2():
    input_content = "15\n1 3 5 2 1 3 2 8 8 6 2 6 11 1 1"
    expected_output = "7"
    run_pie_test_case("../p03816.py", input_content, expected_output)
