from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02411_0():
    input_content = "40 42 -1\n20 30 -1\n0 2 -1\n-1 -1 -1"
    expected_output = "A\nC\nF"
    run_pie_test_case("../p02411.py", input_content, expected_output)


def test_problem_p02411_1():
    input_content = "40 42 -1\n20 30 -1\n0 2 -1\n-1 -1 -1"
    expected_output = "A\nC\nF"
    run_pie_test_case("../p02411.py", input_content, expected_output)
