from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03440_0():
    input_content = "7 5\n1 2 3 4 5 6 7\n3 0\n4 0\n1 2\n1 3\n5 6"
    expected_output = "7"
    run_pie_test_case("../p03440.py", input_content, expected_output)


def test_problem_p03440_1():
    input_content = "1 0\n5"
    expected_output = "0"
    run_pie_test_case("../p03440.py", input_content, expected_output)


def test_problem_p03440_2():
    input_content = "7 5\n1 2 3 4 5 6 7\n3 0\n4 0\n1 2\n1 3\n5 6"
    expected_output = "7"
    run_pie_test_case("../p03440.py", input_content, expected_output)


def test_problem_p03440_3():
    input_content = "5 0\n3 1 4 1 5"
    expected_output = "Impossible"
    run_pie_test_case("../p03440.py", input_content, expected_output)
