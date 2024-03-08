from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03720_0():
    input_content = "4 3\n1 2\n2 3\n1 4"
    expected_output = "2\n2\n1\n1"
    run_pie_test_case("../p03720.py", input_content, expected_output)


def test_problem_p03720_1():
    input_content = "4 3\n1 2\n2 3\n1 4"
    expected_output = "2\n2\n1\n1"
    run_pie_test_case("../p03720.py", input_content, expected_output)


def test_problem_p03720_2():
    input_content = "8 8\n1 2\n3 4\n1 5\n2 8\n3 7\n5 2\n4 1\n6 8"
    expected_output = "3\n3\n2\n2\n2\n1\n1\n2"
    run_pie_test_case("../p03720.py", input_content, expected_output)


def test_problem_p03720_3():
    input_content = "2 5\n1 2\n2 1\n1 2\n2 1\n1 2"
    expected_output = "5\n5"
    run_pie_test_case("../p03720.py", input_content, expected_output)
