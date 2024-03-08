from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03826_0():
    input_content = "3 5 2 7"
    expected_output = "15"
    run_pie_test_case("../p03826.py", input_content, expected_output)


def test_problem_p03826_1():
    input_content = "100 600 200 300"
    expected_output = "60000"
    run_pie_test_case("../p03826.py", input_content, expected_output)


def test_problem_p03826_2():
    input_content = "3 5 2 7"
    expected_output = "15"
    run_pie_test_case("../p03826.py", input_content, expected_output)
