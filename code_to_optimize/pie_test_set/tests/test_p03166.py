from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03166_0():
    input_content = "4 5\n1 2\n1 3\n3 2\n2 4\n3 4"
    expected_output = "3"
    run_pie_test_case("../p03166.py", input_content, expected_output)


def test_problem_p03166_1():
    input_content = "6 3\n2 3\n4 5\n5 6"
    expected_output = "2"
    run_pie_test_case("../p03166.py", input_content, expected_output)


def test_problem_p03166_2():
    input_content = "5 8\n5 3\n2 3\n2 4\n5 2\n5 1\n1 4\n4 3\n1 3"
    expected_output = "3"
    run_pie_test_case("../p03166.py", input_content, expected_output)


def test_problem_p03166_3():
    input_content = "4 5\n1 2\n1 3\n3 2\n2 4\n3 4"
    expected_output = "3"
    run_pie_test_case("../p03166.py", input_content, expected_output)
