from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02537_0():
    input_content = "10 3\n1\n5\n4\n3\n8\n6\n9\n7\n2\n4"
    expected_output = "7"
    run_pie_test_case("../p02537.py", input_content, expected_output)


def test_problem_p02537_1():
    input_content = "10 3\n1\n5\n4\n3\n8\n6\n9\n7\n2\n4"
    expected_output = "7"
    run_pie_test_case("../p02537.py", input_content, expected_output)
