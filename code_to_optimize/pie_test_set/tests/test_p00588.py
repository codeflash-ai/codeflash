from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00588_0():
    input_content = "2\n2\nYNNNNYYY\n4\nNYNNYYNNNYNYYNNN"
    expected_output = "6\n9"
    run_pie_test_case("../p00588.py", input_content, expected_output)


def test_problem_p00588_1():
    input_content = "2\n2\nYNNNNYYY\n4\nNYNNYYNNNYNYYNNN"
    expected_output = "6\n9"
    run_pie_test_case("../p00588.py", input_content, expected_output)
