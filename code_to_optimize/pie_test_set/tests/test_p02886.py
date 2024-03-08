from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02886_0():
    input_content = "3\n3 1 2"
    expected_output = "11"
    run_pie_test_case("../p02886.py", input_content, expected_output)


def test_problem_p02886_1():
    input_content = "3\n3 1 2"
    expected_output = "11"
    run_pie_test_case("../p02886.py", input_content, expected_output)


def test_problem_p02886_2():
    input_content = "7\n5 0 7 8 3 3 2"
    expected_output = "312"
    run_pie_test_case("../p02886.py", input_content, expected_output)
