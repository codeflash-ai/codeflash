from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02734_0():
    input_content = "3 4\n2 2 4"
    expected_output = "5"
    run_pie_test_case("../p02734.py", input_content, expected_output)


def test_problem_p02734_1():
    input_content = "10 10\n3 1 4 1 5 9 2 6 5 3"
    expected_output = "152"
    run_pie_test_case("../p02734.py", input_content, expected_output)


def test_problem_p02734_2():
    input_content = "5 8\n9 9 9 9 9"
    expected_output = "0"
    run_pie_test_case("../p02734.py", input_content, expected_output)


def test_problem_p02734_3():
    input_content = "3 4\n2 2 4"
    expected_output = "5"
    run_pie_test_case("../p02734.py", input_content, expected_output)
