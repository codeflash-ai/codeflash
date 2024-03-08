from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02452_0():
    input_content = "4\n1 2 3 4\n2\n2 4"
    expected_output = "1"
    run_pie_test_case("../p02452.py", input_content, expected_output)


def test_problem_p02452_1():
    input_content = "4\n1 2 3 4\n2\n2 4"
    expected_output = "1"
    run_pie_test_case("../p02452.py", input_content, expected_output)


def test_problem_p02452_2():
    input_content = "4\n1 2 3 4\n3\n1 2 5"
    expected_output = "0"
    run_pie_test_case("../p02452.py", input_content, expected_output)
