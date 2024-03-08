from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03559_0():
    input_content = "2\n1 5\n2 4\n3 6"
    expected_output = "3"
    run_pie_test_case("../p03559.py", input_content, expected_output)


def test_problem_p03559_1():
    input_content = "3\n1 1 1\n2 2 2\n3 3 3"
    expected_output = "27"
    run_pie_test_case("../p03559.py", input_content, expected_output)


def test_problem_p03559_2():
    input_content = "2\n1 5\n2 4\n3 6"
    expected_output = "3"
    run_pie_test_case("../p03559.py", input_content, expected_output)


def test_problem_p03559_3():
    input_content = "6\n3 14 159 2 6 53\n58 9 79 323 84 6\n2643 383 2 79 50 288"
    expected_output = "87"
    run_pie_test_case("../p03559.py", input_content, expected_output)
