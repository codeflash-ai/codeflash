from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03671_0():
    input_content = "700 600 780"
    expected_output = "1300"
    run_pie_test_case("../p03671.py", input_content, expected_output)


def test_problem_p03671_1():
    input_content = "10000 10000 10000"
    expected_output = "20000"
    run_pie_test_case("../p03671.py", input_content, expected_output)


def test_problem_p03671_2():
    input_content = "700 600 780"
    expected_output = "1300"
    run_pie_test_case("../p03671.py", input_content, expected_output)
