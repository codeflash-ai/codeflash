from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00219_0():
    input_content = "15\n2\n6\n7\n0\n1\n9\n8\n7\n3\n8\n9\n4\n8\n2\n2\n3\n9\n1\n5\n0"
    expected_output = "*\n*\n***\n*\n*\n-\n*\n**\n***\n**\n-\n*\n-\n-\n-\n*\n-\n-\n-\n*"
    run_pie_test_case("../p00219.py", input_content, expected_output)


def test_problem_p00219_1():
    input_content = "15\n2\n6\n7\n0\n1\n9\n8\n7\n3\n8\n9\n4\n8\n2\n2\n3\n9\n1\n5\n0"
    expected_output = "*\n*\n***\n*\n*\n-\n*\n**\n***\n**\n-\n*\n-\n-\n-\n*\n-\n-\n-\n*"
    run_pie_test_case("../p00219.py", input_content, expected_output)
