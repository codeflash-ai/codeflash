from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03163_0():
    input_content = "3 8\n3 30\n4 50\n5 60"
    expected_output = "90"
    run_pie_test_case("../p03163.py", input_content, expected_output)
