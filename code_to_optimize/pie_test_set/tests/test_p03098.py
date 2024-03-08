from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03098_0():
    input_content = "3 3\n1 2 3\n3 2 1"
    expected_output = "3 2 1"
    run_pie_test_case("../p03098.py", input_content, expected_output)


def test_problem_p03098_1():
    input_content = "5 5\n4 5 1 2 3\n3 2 1 5 4"
    expected_output = "4 3 2 1 5"
    run_pie_test_case("../p03098.py", input_content, expected_output)


def test_problem_p03098_2():
    input_content = "10 1000000000\n7 10 6 5 4 2 9 1 3 8\n4 1 9 2 3 7 8 10 6 5"
    expected_output = "7 9 4 8 2 5 1 6 10 3"
    run_pie_test_case("../p03098.py", input_content, expected_output)


def test_problem_p03098_3():
    input_content = "3 3\n1 2 3\n3 2 1"
    expected_output = "3 2 1"
    run_pie_test_case("../p03098.py", input_content, expected_output)
