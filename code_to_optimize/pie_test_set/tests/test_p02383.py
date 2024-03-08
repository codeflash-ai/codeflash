from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02383_0():
    input_content = "1 2 4 8 16 32\nSE"
    expected_output = "8"
    run_pie_test_case("../p02383.py", input_content, expected_output)


def test_problem_p02383_1():
    input_content = "1 2 4 8 16 32\nEESWN"
    expected_output = "32"
    run_pie_test_case("../p02383.py", input_content, expected_output)


def test_problem_p02383_2():
    input_content = "1 2 4 8 16 32\nSE"
    expected_output = "8"
    run_pie_test_case("../p02383.py", input_content, expected_output)
