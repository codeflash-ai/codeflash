from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02272_0():
    input_content = "10\n8 5 9 2 6 3 7 1 10 4"
    expected_output = "1 2 3 4 5 6 7 8 9 10\n34"
    run_pie_test_case("../p02272.py", input_content, expected_output)


def test_problem_p02272_1():
    input_content = "10\n8 5 9 2 6 3 7 1 10 4"
    expected_output = "1 2 3 4 5 6 7 8 9 10\n34"
    run_pie_test_case("../p02272.py", input_content, expected_output)
