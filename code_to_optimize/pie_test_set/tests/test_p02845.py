from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02845_0():
    input_content = "6\n0 1 2 3 4 5"
    expected_output = "3"
    run_pie_test_case("../p02845.py", input_content, expected_output)


def test_problem_p02845_1():
    input_content = "54\n0 0 1 0 1 2 1 2 3 2 3 3 4 4 5 4 6 5 7 8 5 6 6 7 7 8 8 9 9 10 10 11 9 12 10 13 14 11 11 12 12 13 13 14 14 15 15 15 16 16 16 17 17 17"
    expected_output = "115295190"
    run_pie_test_case("../p02845.py", input_content, expected_output)


def test_problem_p02845_2():
    input_content = "3\n0 0 0"
    expected_output = "6"
    run_pie_test_case("../p02845.py", input_content, expected_output)


def test_problem_p02845_3():
    input_content = "6\n0 1 2 3 4 5"
    expected_output = "3"
    run_pie_test_case("../p02845.py", input_content, expected_output)
