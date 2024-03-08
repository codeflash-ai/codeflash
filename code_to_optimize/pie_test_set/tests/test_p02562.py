from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02562_0():
    input_content = "3 1\n5 3 2\n1 4 8\n7 6 9"
    expected_output = "19\nX..\n..X\n.X."
    run_pie_test_case("../p02562.py", input_content, expected_output)


def test_problem_p02562_1():
    input_content = "3 2\n10 10 1\n10 10 1\n1 1 10"
    expected_output = "50\nXX.\nXX.\n..X"
    run_pie_test_case("../p02562.py", input_content, expected_output)


def test_problem_p02562_2():
    input_content = "3 1\n5 3 2\n1 4 8\n7 6 9"
    expected_output = "19\nX..\n..X\n.X."
    run_pie_test_case("../p02562.py", input_content, expected_output)
