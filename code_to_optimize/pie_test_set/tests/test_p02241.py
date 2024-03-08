from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02241_0():
    input_content = "5\n -1 2 3 1 -1\n 2 -1 -1 4 -1\n 3 -1 -1 1 1\n 1 4 1 -1 3\n -1 -1 1 3 -1"
    expected_output = "5"
    run_pie_test_case("../p02241.py", input_content, expected_output)
