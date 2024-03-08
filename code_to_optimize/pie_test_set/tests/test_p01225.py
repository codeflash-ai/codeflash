from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01225_0():
    input_content = "5\n1 2 3 3 4 5 7 7 7\nR R R R R R G G G\n1 2 2 3 4 4 4 4 5\nR R R R R R R R R\n1 2 3 4 4 4 5 5 5\nR R B R R R R R R\n1 1 1 3 4 5 6 6 6\nR R B G G G R R R\n2 2 2 3 3 3 1 1 1\nR G B R G B R G B"
    expected_output = "1\n0\n0\n0\n1"
    run_pie_test_case("../p01225.py", input_content, expected_output)


def test_problem_p01225_1():
    input_content = "5\n1 2 3 3 4 5 7 7 7\nR R R R R R G G G\n1 2 2 3 4 4 4 4 5\nR R R R R R R R R\n1 2 3 4 4 4 5 5 5\nR R B R R R R R R\n1 1 1 3 4 5 6 6 6\nR R B G G G R R R\n2 2 2 3 3 3 1 1 1\nR G B R G B R G B"
    expected_output = "1\n0\n0\n0\n1"
    run_pie_test_case("../p01225.py", input_content, expected_output)
