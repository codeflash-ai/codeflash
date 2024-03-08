from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02647_0():
    input_content = "5 1\n1 0 0 1 0"
    expected_output = "1 2 2 1 2"
    run_pie_test_case("../p02647.py", input_content, expected_output)


def test_problem_p02647_1():
    input_content = "5 1\n1 0 0 1 0"
    expected_output = "1 2 2 1 2"
    run_pie_test_case("../p02647.py", input_content, expected_output)


def test_problem_p02647_2():
    input_content = "5 2\n1 0 0 1 0"
    expected_output = "3 3 4 4 3"
    run_pie_test_case("../p02647.py", input_content, expected_output)
