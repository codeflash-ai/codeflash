from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00147_0():
    input_content = "5\n6\n7\n8"
    expected_output = "0\n14\n9\n4"
    run_pie_test_case("../p00147.py", input_content, expected_output)


def test_problem_p00147_1():
    input_content = "5\n6\n7\n8"
    expected_output = "0\n14\n9\n4"
    run_pie_test_case("../p00147.py", input_content, expected_output)
