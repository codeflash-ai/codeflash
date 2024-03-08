from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02888_0():
    input_content = "4\n3 4 2 1"
    expected_output = "1"
    run_pie_test_case("../p02888.py", input_content, expected_output)


def test_problem_p02888_1():
    input_content = "4\n3 4 2 1"
    expected_output = "1"
    run_pie_test_case("../p02888.py", input_content, expected_output)


def test_problem_p02888_2():
    input_content = "7\n218 786 704 233 645 728 389"
    expected_output = "23"
    run_pie_test_case("../p02888.py", input_content, expected_output)


def test_problem_p02888_3():
    input_content = "3\n1 1000 1"
    expected_output = "0"
    run_pie_test_case("../p02888.py", input_content, expected_output)
