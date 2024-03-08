from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02255_0():
    input_content = "6\n5 2 4 6 1 3"
    expected_output = "5 2 4 6 1 3\n2 5 4 6 1 3\n2 4 5 6 1 3\n2 4 5 6 1 3\n1 2 4 5 6 3\n1 2 3 4 5 6"
    run_pie_test_case("../p02255.py", input_content, expected_output)


def test_problem_p02255_1():
    input_content = "3\n1 2 3"
    expected_output = "1 2 3\n1 2 3\n1 2 3"
    run_pie_test_case("../p02255.py", input_content, expected_output)


def test_problem_p02255_2():
    input_content = "6\n5 2 4 6 1 3"
    expected_output = "5 2 4 6 1 3\n2 5 4 6 1 3\n2 4 5 6 1 3\n2 4 5 6 1 3\n1 2 4 5 6 3\n1 2 3 4 5 6"
    run_pie_test_case("../p02255.py", input_content, expected_output)
