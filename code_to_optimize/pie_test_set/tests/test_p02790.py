from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02790_0():
    input_content = "4 3"
    expected_output = "3333"
    run_pie_test_case("../p02790.py", input_content, expected_output)


def test_problem_p02790_1():
    input_content = "7 7"
    expected_output = "7777777"
    run_pie_test_case("../p02790.py", input_content, expected_output)


def test_problem_p02790_2():
    input_content = "4 3"
    expected_output = "3333"
    run_pie_test_case("../p02790.py", input_content, expected_output)
