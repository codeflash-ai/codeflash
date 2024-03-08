from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03619_0():
    input_content = "1 1 6 5\n3\n3 2\n5 3\n2 4"
    expected_output = "891.415926535897938"
    run_pie_test_case("../p03619.py", input_content, expected_output)


def test_problem_p03619_1():
    input_content = "4 2 2 2\n3\n3 2\n5 3\n2 4"
    expected_output = "211.415926535897938"
    run_pie_test_case("../p03619.py", input_content, expected_output)


def test_problem_p03619_2():
    input_content = "1 1 6 5\n3\n3 2\n5 3\n2 4"
    expected_output = "891.415926535897938"
    run_pie_test_case("../p03619.py", input_content, expected_output)


def test_problem_p03619_3():
    input_content = "3 5 6 4\n3\n3 2\n5 3\n2 4"
    expected_output = "400.000000000000000"
    run_pie_test_case("../p03619.py", input_content, expected_output)
