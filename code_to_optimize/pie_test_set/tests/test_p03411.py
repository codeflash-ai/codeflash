from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03411_0():
    input_content = "3\n2 0\n3 1\n1 3\n4 2\n0 4\n5 5"
    expected_output = "2"
    run_pie_test_case("../p03411.py", input_content, expected_output)
