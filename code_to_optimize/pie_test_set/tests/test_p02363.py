from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02363_0():
    input_content = "4 6\n0 1 1\n0 2 5\n1 2 2\n1 3 4\n2 3 1\n3 2 7"
    expected_output = "0 1 3 4\nINF 0 2 3\nINF INF 0 1\nINF INF 7 0"
    run_pie_test_case("../p02363.py", input_content, expected_output)


def test_problem_p02363_1():
    input_content = "4 6\n0 1 1\n0 2 -5\n1 2 2\n1 3 4\n2 3 1\n3 2 7"
    expected_output = "0 1 -5 -4\nINF 0 2 3\nINF INF 0 1\nINF INF 7 0"
    run_pie_test_case("../p02363.py", input_content, expected_output)


def test_problem_p02363_2():
    input_content = "4 6\n0 1 1\n0 2 5\n1 2 2\n1 3 4\n2 3 1\n3 2 -7"
    expected_output = "NEGATIVE CYCLE"
    run_pie_test_case("../p02363.py", input_content, expected_output)


def test_problem_p02363_3():
    input_content = "4 6\n0 1 1\n0 2 5\n1 2 2\n1 3 4\n2 3 1\n3 2 7"
    expected_output = "0 1 3 4\nINF 0 2 3\nINF INF 0 1\nINF INF 7 0"
    run_pie_test_case("../p02363.py", input_content, expected_output)
