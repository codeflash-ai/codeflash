from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03766_0():
    input_content = "2"
    expected_output = "4"
    run_pie_test_case("../p03766.py", input_content, expected_output)


def test_problem_p03766_1():
    input_content = "654321"
    expected_output = "968545283"
    run_pie_test_case("../p03766.py", input_content, expected_output)


def test_problem_p03766_2():
    input_content = "2"
    expected_output = "4"
    run_pie_test_case("../p03766.py", input_content, expected_output)
