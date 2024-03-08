from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03354_0():
    input_content = "5 2\n5 3 1 4 2\n1 3\n5 4"
    expected_output = "2"
    run_pie_test_case("../p03354.py", input_content, expected_output)


def test_problem_p03354_1():
    input_content = "5 1\n1 2 3 4 5\n1 5"
    expected_output = "5"
    run_pie_test_case("../p03354.py", input_content, expected_output)


def test_problem_p03354_2():
    input_content = "10 8\n5 3 6 8 7 10 9 1 2 4\n3 1\n4 1\n5 9\n2 5\n6 5\n3 5\n8 9\n7 9"
    expected_output = "8"
    run_pie_test_case("../p03354.py", input_content, expected_output)


def test_problem_p03354_3():
    input_content = "3 2\n3 2 1\n1 2\n2 3"
    expected_output = "3"
    run_pie_test_case("../p03354.py", input_content, expected_output)


def test_problem_p03354_4():
    input_content = "5 2\n5 3 1 4 2\n1 3\n5 4"
    expected_output = "2"
    run_pie_test_case("../p03354.py", input_content, expected_output)
