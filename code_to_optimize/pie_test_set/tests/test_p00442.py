from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00442_0():
    input_content = "4\n5\n1 2\n3 1\n3 2\n3 4\n4 1"
    expected_output = "3\n4\n1\n2\n0"
    run_pie_test_case("../p00442.py", input_content, expected_output)
