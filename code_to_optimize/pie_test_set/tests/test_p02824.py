from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02824_0():
    input_content = "6 1 2 2\n2 1 1 3 0 2"
    expected_output = "5"
    run_pie_test_case("../p02824.py", input_content, expected_output)


def test_problem_p02824_1():
    input_content = "6 1 2 2\n2 1 1 3 0 2"
    expected_output = "5"
    run_pie_test_case("../p02824.py", input_content, expected_output)


def test_problem_p02824_2():
    input_content = "10 4 8 5\n7 2 3 6 1 6 5 4 6 5"
    expected_output = "8"
    run_pie_test_case("../p02824.py", input_content, expected_output)


def test_problem_p02824_3():
    input_content = "6 1 5 2\n2 1 1 3 0 2"
    expected_output = "3"
    run_pie_test_case("../p02824.py", input_content, expected_output)
