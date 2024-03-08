from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03295_0():
    input_content = "5 2\n1 4\n2 5"
    expected_output = "1"
    run_pie_test_case("../p03295.py", input_content, expected_output)


def test_problem_p03295_1():
    input_content = "5 10\n1 2\n1 3\n1 4\n1 5\n2 3\n2 4\n2 5\n3 4\n3 5\n4 5"
    expected_output = "4"
    run_pie_test_case("../p03295.py", input_content, expected_output)


def test_problem_p03295_2():
    input_content = "5 2\n1 4\n2 5"
    expected_output = "1"
    run_pie_test_case("../p03295.py", input_content, expected_output)


def test_problem_p03295_3():
    input_content = "9 5\n1 8\n2 7\n3 5\n4 6\n7 9"
    expected_output = "2"
    run_pie_test_case("../p03295.py", input_content, expected_output)
