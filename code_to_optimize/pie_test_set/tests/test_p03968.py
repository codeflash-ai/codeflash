from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03968_0():
    input_content = "6\n0 1 2 3\n0 4 6 1\n1 6 7 2\n2 7 5 3\n6 4 5 7\n4 0 3 5"
    expected_output = "1"
    run_pie_test_case("../p03968.py", input_content, expected_output)


def test_problem_p03968_1():
    input_content = "6\n0 1 2 3\n0 4 6 1\n1 6 7 2\n2 7 5 3\n6 4 5 7\n4 0 3 5"
    expected_output = "1"
    run_pie_test_case("../p03968.py", input_content, expected_output)


def test_problem_p03968_2():
    input_content = "6\n0 0 0 0\n0 0 0 0\n0 0 0 0\n0 0 0 0\n0 0 0 0\n0 0 0 0"
    expected_output = "122880"
    run_pie_test_case("../p03968.py", input_content, expected_output)


def test_problem_p03968_3():
    input_content = "8\n0 0 0 0\n0 0 1 1\n0 1 0 1\n0 1 1 0\n1 0 0 1\n1 0 1 0\n1 1 0 0\n1 1 1 1"
    expected_output = "144"
    run_pie_test_case("../p03968.py", input_content, expected_output)
