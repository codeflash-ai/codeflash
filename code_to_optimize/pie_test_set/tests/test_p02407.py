from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02407_0():
    input_content = "5\n1 2 3 4 5"
    expected_output = "5 4 3 2 1"
    run_pie_test_case("../p02407.py", input_content, expected_output)


def test_problem_p02407_1():
    input_content = "8\n3 3 4 4 5 8 7 9"
    expected_output = "9 7 8 5 4 4 3 3"
    run_pie_test_case("../p02407.py", input_content, expected_output)


def test_problem_p02407_2():
    input_content = "5\n1 2 3 4 5"
    expected_output = "5 4 3 2 1"
    run_pie_test_case("../p02407.py", input_content, expected_output)
