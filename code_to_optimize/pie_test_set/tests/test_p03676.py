from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03676_0():
    input_content = "3\n1 2 1 3"
    expected_output = "3\n5\n4\n1"
    run_pie_test_case("../p03676.py", input_content, expected_output)
