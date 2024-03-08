from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02850_0():
    input_content = "3\n1 2\n2 3"
    expected_output = "2\n1\n2"
    run_pie_test_case("../p02850.py", input_content, expected_output)


def test_problem_p02850_1():
    input_content = "3\n1 2\n2 3"
    expected_output = "2\n1\n2"
    run_pie_test_case("../p02850.py", input_content, expected_output)


def test_problem_p02850_2():
    input_content = "6\n1 2\n1 3\n1 4\n1 5\n1 6"
    expected_output = "5\n1\n2\n3\n4\n5"
    run_pie_test_case("../p02850.py", input_content, expected_output)


def test_problem_p02850_3():
    input_content = "8\n1 2\n2 3\n2 4\n2 5\n4 7\n5 6\n6 8"
    expected_output = "4\n1\n2\n3\n4\n1\n1\n2"
    run_pie_test_case("../p02850.py", input_content, expected_output)
