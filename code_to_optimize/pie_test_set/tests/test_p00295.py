from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00295_0():
    input_content = "4\n1 1 1 1 1 1 1 1 1 2 2 2 4 4 4 6 6 6 5 5 5 3 3 3 3 3 3 3 3 3\n3 3 3 1 1 1 1 1 1 2 2 2 4 4 6 4 6 6 5 5 5 3 3 3 3 3 3 1 1 1\n3 3 3 1 1 3 1 1 1 2 2 5 6 4 4 4 6 6 2 5 5 3 3 3 1 3 3 1 1 1\n1 3 1 3 1 3 3 1 3 2 2 2 6 4 4 6 6 4 5 5 5 1 3 1 1 3 1 3 1 3"
    expected_output = "0\n1\n2\n7"
    run_pie_test_case("../p00295.py", input_content, expected_output)
