from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00535_0():
    input_content = "5 6\n......\n.939..\n.3428.\n.9393.\n......"
    expected_output = "3"
    run_pie_test_case("../p00535.py", input_content, expected_output)
