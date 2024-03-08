from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03320_0():
    input_content = "10"
    expected_output = "1\n2\n3\n4\n5\n6\n7\n8\n9\n19"
    run_pie_test_case("../p03320.py", input_content, expected_output)
