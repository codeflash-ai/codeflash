from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03363_0():
    input_content = "6\n1 3 -4 2 2 -2"
    expected_output = "3"
    run_pie_test_case("../p03363.py", input_content, expected_output)


def test_problem_p03363_1():
    input_content = "7\n1 -1 1 -1 1 -1 1"
    expected_output = "12"
    run_pie_test_case("../p03363.py", input_content, expected_output)


def test_problem_p03363_2():
    input_content = "5\n1 -2 3 -4 5"
    expected_output = "0"
    run_pie_test_case("../p03363.py", input_content, expected_output)


def test_problem_p03363_3():
    input_content = "6\n1 3 -4 2 2 -2"
    expected_output = "3"
    run_pie_test_case("../p03363.py", input_content, expected_output)
