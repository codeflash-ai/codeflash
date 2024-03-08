from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02362_0():
    input_content = "4 5 0\n0 1 2\n0 2 3\n1 2 -5\n1 3 1\n2 3 2"
    expected_output = "0\n2\n-3\n-1"
    run_pie_test_case("../p02362.py", input_content, expected_output)


def test_problem_p02362_1():
    input_content = "4 5 1\n0 1 2\n0 2 3\n1 2 -5\n1 3 1\n2 3 2"
    expected_output = "INF\n0\n-5\n-3"
    run_pie_test_case("../p02362.py", input_content, expected_output)


def test_problem_p02362_2():
    input_content = "4 5 0\n0 1 2\n0 2 3\n1 2 -5\n1 3 1\n2 3 2"
    expected_output = "0\n2\n-3\n-1"
    run_pie_test_case("../p02362.py", input_content, expected_output)


def test_problem_p02362_3():
    input_content = "4 6 0\n0 1 2\n0 2 3\n1 2 -5\n1 3 1\n2 3 2\n3 1 0"
    expected_output = "NEGATIVE CYCLE"
    run_pie_test_case("../p02362.py", input_content, expected_output)
