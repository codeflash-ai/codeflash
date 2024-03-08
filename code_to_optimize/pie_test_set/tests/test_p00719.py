from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00719_0():
    input_content = "3 4 3 1 4\n3 1 2\n1 2 10\n2 3 30\n3 4 20\n2 4 4 2 1\n3 1\n2 3 3\n1 3 3\n4 1 2\n4 2 5\n2 4 3 4 1\n5 5\n1 2 10\n2 3 10\n3 4 10\n1 2 0 1 2\n1\n8 5 10 1 5\n2 7 1 8 4 5 6 3\n1 2 5\n2 3 4\n3 4 7\n4 5 3\n1 3 25\n2 4 23\n3 5 22\n1 4 45\n2 5 51\n1 5 99\n0 0 0 0 0"
    expected_output = "30.000\n3.667\nImpossible\nImpossible\n2.856"
    run_pie_test_case("../p00719.py", input_content, expected_output)


def test_problem_p00719_1():
    input_content = "3 4 3 1 4\n3 1 2\n1 2 10\n2 3 30\n3 4 20\n2 4 4 2 1\n3 1\n2 3 3\n1 3 3\n4 1 2\n4 2 5\n2 4 3 4 1\n5 5\n1 2 10\n2 3 10\n3 4 10\n1 2 0 1 2\n1\n8 5 10 1 5\n2 7 1 8 4 5 6 3\n1 2 5\n2 3 4\n3 4 7\n4 5 3\n1 3 25\n2 4 23\n3 5 22\n1 4 45\n2 5 51\n1 5 99\n0 0 0 0 0"
    expected_output = "30.000\n3.667\nImpossible\nImpossible\n2.856"
    run_pie_test_case("../p00719.py", input_content, expected_output)
