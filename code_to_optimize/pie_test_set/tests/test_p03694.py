from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03694_0():
    input_content = "4\n2 3 7 9"
    expected_output = "7"
    run_pie_test_case("../p03694.py", input_content, expected_output)


def test_problem_p03694_1():
    input_content = "4\n2 3 7 9"
    expected_output = "7"
    run_pie_test_case("../p03694.py", input_content, expected_output)


def test_problem_p03694_2():
    input_content = "8\n3 1 4 1 5 9 2 6"
    expected_output = "8"
    run_pie_test_case("../p03694.py", input_content, expected_output)
