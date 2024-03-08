from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00277_0():
    input_content = "3 4 600\n3 100 5\n1 200 10\n2 400 20\n3 500 20"
    expected_output = "1"
    run_pie_test_case("../p00277.py", input_content, expected_output)


def test_problem_p00277_1():
    input_content = "3 4 600\n3 100 5\n1 200 10\n2 400 20\n3 500 20"
    expected_output = "1"
    run_pie_test_case("../p00277.py", input_content, expected_output)
