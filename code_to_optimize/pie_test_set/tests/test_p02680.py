from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02680_0():
    input_content = (
        "5 6\n1 2 0\n0 1 1\n0 2 2\n-3 4 -1\n-2 6 3\n1 0 1\n0 1 2\n2 0 2\n-1 -4 5\n3 -2 4\n1 2 4"
    )
    expected_output = "13"
    run_pie_test_case("../p02680.py", input_content, expected_output)


def test_problem_p02680_1():
    input_content = "6 1\n-3 -1 -2\n-3 -1 1\n-2 -1 2\n1 4 -2\n1 4 -1\n1 4 1\n3 1 4"
    expected_output = "INF"
    run_pie_test_case("../p02680.py", input_content, expected_output)


def test_problem_p02680_2():
    input_content = (
        "5 6\n1 2 0\n0 1 1\n0 2 2\n-3 4 -1\n-2 6 3\n1 0 1\n0 1 2\n2 0 2\n-1 -4 5\n3 -2 4\n1 2 4"
    )
    expected_output = "13"
    run_pie_test_case("../p02680.py", input_content, expected_output)
