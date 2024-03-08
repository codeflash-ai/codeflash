from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00048_0():
    input_content = "60.2\n70.2\n48.0\n80.2"
    expected_output = "light welter\nlight middle\nlight fly\nmiddle"
    run_pie_test_case("../p00048.py", input_content, expected_output)
