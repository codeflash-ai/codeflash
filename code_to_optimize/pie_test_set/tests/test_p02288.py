from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02288_0():
    input_content = "10\n4 1 3 2 16 9 10 14 8 7"
    expected_output = "16 14 10 8 7 9 3 2 4 1"
    run_pie_test_case("../p02288.py", input_content, expected_output)


def test_problem_p02288_1():
    input_content = "10\n4 1 3 2 16 9 10 14 8 7"
    expected_output = "16 14 10 8 7 9 3 2 4 1"
    run_pie_test_case("../p02288.py", input_content, expected_output)
