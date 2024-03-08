from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03321_0():
    input_content = "5 5\n1 2\n1 3\n3 4\n3 5\n4 5"
    expected_output = "4"
    run_pie_test_case("../p03321.py", input_content, expected_output)


def test_problem_p03321_1():
    input_content = "5 5\n1 2\n1 3\n3 4\n3 5\n4 5"
    expected_output = "4"
    run_pie_test_case("../p03321.py", input_content, expected_output)


def test_problem_p03321_2():
    input_content = "5 1\n1 2"
    expected_output = "-1"
    run_pie_test_case("../p03321.py", input_content, expected_output)


def test_problem_p03321_3():
    input_content = "10 39\n7 2\n7 1\n5 6\n5 8\n9 10\n2 8\n8 7\n3 10\n10 1\n8 10\n2 3\n7 4\n3 9\n4 10\n3 4\n6 1\n6 7\n9 5\n9 7\n6 9\n9 4\n4 6\n7 5\n8 3\n2 5\n9 2\n10 7\n8 6\n8 9\n7 3\n5 3\n4 5\n6 3\n2 10\n5 10\n4 2\n6 2\n8 4\n10 6"
    expected_output = "21"
    run_pie_test_case("../p03321.py", input_content, expected_output)


def test_problem_p03321_4():
    input_content = "4 3\n1 2\n1 3\n2 3"
    expected_output = "3"
    run_pie_test_case("../p03321.py", input_content, expected_output)
