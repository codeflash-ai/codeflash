from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03175_0():
    input_content = "3\n1 2\n2 3"
    expected_output = "5"
    run_pie_test_case("../p03175.py", input_content, expected_output)


def test_problem_p03175_1():
    input_content = "3\n1 2\n2 3"
    expected_output = "5"
    run_pie_test_case("../p03175.py", input_content, expected_output)


def test_problem_p03175_2():
    input_content = "10\n8 5\n10 8\n6 5\n1 5\n4 8\n2 10\n3 6\n9 2\n1 7"
    expected_output = "157"
    run_pie_test_case("../p03175.py", input_content, expected_output)


def test_problem_p03175_3():
    input_content = "1"
    expected_output = "2"
    run_pie_test_case("../p03175.py", input_content, expected_output)


def test_problem_p03175_4():
    input_content = "4\n1 2\n1 3\n1 4"
    expected_output = "9"
    run_pie_test_case("../p03175.py", input_content, expected_output)
