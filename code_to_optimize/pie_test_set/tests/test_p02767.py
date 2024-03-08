from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02767_0():
    input_content = "2\n1 4"
    expected_output = "5"
    run_pie_test_case("../p02767.py", input_content, expected_output)


def test_problem_p02767_1():
    input_content = "7\n14 14 2 13 56 2 37"
    expected_output = "2354"
    run_pie_test_case("../p02767.py", input_content, expected_output)


def test_problem_p02767_2():
    input_content = "2\n1 4"
    expected_output = "5"
    run_pie_test_case("../p02767.py", input_content, expected_output)
