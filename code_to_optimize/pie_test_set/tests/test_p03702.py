from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03702_0():
    input_content = "4 5 3\n8\n7\n4\n2"
    expected_output = "2"
    run_pie_test_case("../p03702.py", input_content, expected_output)
