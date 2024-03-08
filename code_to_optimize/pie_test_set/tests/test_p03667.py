from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03667_0():
    input_content = "5 3\n1 1 3 4 5\n1 2\n2 5\n5 4"
    expected_output = "0\n1\n1"
    run_pie_test_case("../p03667.py", input_content, expected_output)


def test_problem_p03667_1():
    input_content = "5 3\n1 1 3 4 5\n1 2\n2 5\n5 4"
    expected_output = "0\n1\n1"
    run_pie_test_case("../p03667.py", input_content, expected_output)


def test_problem_p03667_2():
    input_content = "4 4\n4 4 4 4\n4 1\n3 1\n1 1\n2 1"
    expected_output = "0\n1\n2\n3"
    run_pie_test_case("../p03667.py", input_content, expected_output)


def test_problem_p03667_3():
    input_content = "10 10\n8 7 2 9 10 6 6 5 5 4\n8 1\n6 3\n6 2\n7 10\n9 7\n9 9\n2 4\n8 1\n1 8\n7 7"
    expected_output = "1\n0\n1\n2\n2\n3\n3\n3\n3\n2"
    run_pie_test_case("../p03667.py", input_content, expected_output)
