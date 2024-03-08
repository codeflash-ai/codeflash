from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03453_0():
    input_content = "4 4\n1 3\n1 2 1\n2 3 1\n3 4 1\n4 1 1"
    expected_output = "2"
    run_pie_test_case("../p03453.py", input_content, expected_output)


def test_problem_p03453_1():
    input_content = "4 4\n1 3\n1 2 1\n2 3 1\n3 4 1\n4 1 1"
    expected_output = "2"
    run_pie_test_case("../p03453.py", input_content, expected_output)


def test_problem_p03453_2():
    input_content = "3 3\n1 3\n1 2 1\n2 3 1\n3 1 2"
    expected_output = "2"
    run_pie_test_case("../p03453.py", input_content, expected_output)


def test_problem_p03453_3():
    input_content = "8 13\n4 2\n7 3 9\n6 2 3\n1 6 4\n7 6 9\n3 8 9\n1 2 2\n2 8 12\n8 6 9\n2 5 5\n4 2 18\n5 3 7\n5 1 515371567\n4 8 6"
    expected_output = "6"
    run_pie_test_case("../p03453.py", input_content, expected_output)
