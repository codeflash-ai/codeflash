from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02245_0():
    input_content = "1 3 0\n4 2 5\n7 8 6"
    expected_output = "4"
    run_pie_test_case("../p02245.py", input_content, expected_output)


def test_problem_p02245_1():
    input_content = "1 3 0\n4 2 5\n7 8 6"
    expected_output = "4"
    run_pie_test_case("../p02245.py", input_content, expected_output)
