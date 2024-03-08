from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00479_0():
    input_content = "11\n4\n5 2\n9 7\n4 4\n3 9"
    expected_output = "2\n3\n1\n3"
    run_pie_test_case("../p00479.py", input_content, expected_output)
