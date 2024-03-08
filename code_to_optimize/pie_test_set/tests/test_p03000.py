from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03000_0():
    input_content = "3 6\n3 4 5"
    expected_output = "2"
    run_pie_test_case("../p03000.py", input_content, expected_output)


def test_problem_p03000_1():
    input_content = "4 9\n3 3 3 3"
    expected_output = "4"
    run_pie_test_case("../p03000.py", input_content, expected_output)


def test_problem_p03000_2():
    input_content = "3 6\n3 4 5"
    expected_output = "2"
    run_pie_test_case("../p03000.py", input_content, expected_output)
