from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00462_0():
    input_content = "8\n3\n2\n3\n1\n4\n6\n20\n4\n4\n12\n8\n16\n7\n7\n11\n8\n0"
    expected_output = "3\n3"
    run_pie_test_case("../p00462.py", input_content, expected_output)


def test_problem_p00462_1():
    input_content = "8\n3\n2\n3\n1\n4\n6\n20\n4\n4\n12\n8\n16\n7\n7\n11\n8\n0"
    expected_output = "3\n3"
    run_pie_test_case("../p00462.py", input_content, expected_output)
