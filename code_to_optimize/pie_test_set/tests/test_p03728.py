from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03728_0():
    input_content = "5\n3 5 1 2 4"
    expected_output = "3"
    run_pie_test_case("../p03728.py", input_content, expected_output)


def test_problem_p03728_1():
    input_content = "10\n2 10 5 7 3 6 4 9 8 1"
    expected_output = "6"
    run_pie_test_case("../p03728.py", input_content, expected_output)


def test_problem_p03728_2():
    input_content = "5\n5 4 3 2 1"
    expected_output = "4"
    run_pie_test_case("../p03728.py", input_content, expected_output)


def test_problem_p03728_3():
    input_content = "5\n3 5 1 2 4"
    expected_output = "3"
    run_pie_test_case("../p03728.py", input_content, expected_output)
