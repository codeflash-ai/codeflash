from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00028_0():
    input_content = "5\n6\n3\n5\n8\n7\n5\n3\n9\n7\n3\n4"
    expected_output = "3\n5"
    run_pie_test_case("../p00028.py", input_content, expected_output)


def test_problem_p00028_1():
    input_content = "5\n6\n3\n5\n8\n7\n5\n3\n9\n7\n3\n4"
    expected_output = "3\n5"
    run_pie_test_case("../p00028.py", input_content, expected_output)
