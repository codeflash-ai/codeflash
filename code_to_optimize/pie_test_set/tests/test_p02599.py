from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02599_0():
    input_content = "4 3\n1 2 1 3\n1 3\n2 4\n3 3"
    expected_output = "2\n3\n1"
    run_pie_test_case("../p02599.py", input_content, expected_output)


def test_problem_p02599_1():
    input_content = "4 3\n1 2 1 3\n1 3\n2 4\n3 3"
    expected_output = "2\n3\n1"
    run_pie_test_case("../p02599.py", input_content, expected_output)


def test_problem_p02599_2():
    input_content = "10 10\n2 5 6 5 2 1 7 9 7 2\n5 5\n2 4\n6 7\n2 2\n7 8\n7 9\n1 8\n6 9\n8 10\n6 8"
    expected_output = "1\n2\n2\n1\n2\n2\n6\n3\n3\n3"
    run_pie_test_case("../p02599.py", input_content, expected_output)
