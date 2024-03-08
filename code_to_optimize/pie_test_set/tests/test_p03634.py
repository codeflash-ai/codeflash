from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03634_0():
    input_content = "5\n1 2 1\n1 3 1\n2 4 1\n3 5 1\n3 1\n2 4\n2 3\n4 5"
    expected_output = "3\n2\n4"
    run_pie_test_case("../p03634.py", input_content, expected_output)


def test_problem_p03634_1():
    input_content = "5\n1 2 1\n1 3 1\n2 4 1\n3 5 1\n3 1\n2 4\n2 3\n4 5"
    expected_output = "3\n2\n4"
    run_pie_test_case("../p03634.py", input_content, expected_output)


def test_problem_p03634_2():
    input_content = "7\n1 2 1\n1 3 3\n1 4 5\n1 5 7\n1 6 9\n1 7 11\n3 2\n1 3\n4 5\n6 7"
    expected_output = "5\n14\n22"
    run_pie_test_case("../p03634.py", input_content, expected_output)


def test_problem_p03634_3():
    input_content = "10\n1 2 1000000000\n2 3 1000000000\n3 4 1000000000\n4 5 1000000000\n5 6 1000000000\n6 7 1000000000\n7 8 1000000000\n8 9 1000000000\n9 10 1000000000\n1 1\n9 10"
    expected_output = "17000000000"
    run_pie_test_case("../p03634.py", input_content, expected_output)
