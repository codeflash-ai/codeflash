from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00767_0():
    input_content = "1 2\n1 3\n2 3\n1 4\n2 4\n5 6\n1 8\n4 7\n98 100\n99 100\n0 0"
    expected_output = "1 3\n2 3\n1 4\n2 4\n3 4\n1 8\n4 7\n2 8\n3 140\n89 109"
    run_pie_test_case("../p00767.py", input_content, expected_output)
