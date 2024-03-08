from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02986_0():
    input_content = "5 3\n1 2 1 10\n1 3 2 20\n2 4 4 30\n5 2 1 40\n1 100 1 4\n1 100 1 5\n3 1000 3 4"
    expected_output = "130\n200\n60"
    run_pie_test_case("../p02986.py", input_content, expected_output)


def test_problem_p02986_1():
    input_content = "5 3\n1 2 1 10\n1 3 2 20\n2 4 4 30\n5 2 1 40\n1 100 1 4\n1 100 1 5\n3 1000 3 4"
    expected_output = "130\n200\n60"
    run_pie_test_case("../p02986.py", input_content, expected_output)
