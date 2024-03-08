from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01525_0():
    input_content = "3 11\n1 0 2\n4 1 3\n7 2 4\n0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10"
    expected_output = "1\n3\n4\n0\n1\n3\n5\n7\n11\n19\n29\n46\n47\n48"
    run_pie_test_case("../p01525.py", input_content, expected_output)
