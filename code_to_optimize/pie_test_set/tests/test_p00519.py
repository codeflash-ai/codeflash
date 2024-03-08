from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00519_0():
    input_content = "6 6\n400 2\n200 1\n600 3\n1000 1\n300 5\n700 4\n1 2\n2 3\n3 6\n4 6\n1 5\n2 4"
    expected_output = "700"
    run_pie_test_case("../p00519.py", input_content, expected_output)


def test_problem_p00519_1():
    input_content = "6 6\n400 2\n200 1\n600 3\n1000 1\n300 5\n700 4\n1 2\n2 3\n3 6\n4 6\n1 5\n2 4"
    expected_output = "700"
    run_pie_test_case("../p00519.py", input_content, expected_output)


def test_problem_p00519_2():
    input_content = "None"
    expected_output = "None"
    run_pie_test_case("../p00519.py", input_content, expected_output)
