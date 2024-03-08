from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02974_0():
    input_content = "3 2"
    expected_output = "2"
    run_pie_test_case("../p02974.py", input_content, expected_output)


def test_problem_p02974_1():
    input_content = "39 14"
    expected_output = "74764168"
    run_pie_test_case("../p02974.py", input_content, expected_output)


def test_problem_p02974_2():
    input_content = "3 2"
    expected_output = "2"
    run_pie_test_case("../p02974.py", input_content, expected_output)
