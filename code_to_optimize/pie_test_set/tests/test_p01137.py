from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01137_0():
    input_content = "1\n2\n4\n27\n300\n1250\n0"
    expected_output = "1\n2\n2\n3\n18\n44"
    run_pie_test_case("../p01137.py", input_content, expected_output)


def test_problem_p01137_1():
    input_content = "1\n2\n4\n27\n300\n1250\n0"
    expected_output = "1\n2\n2\n3\n18\n44"
    run_pie_test_case("../p01137.py", input_content, expected_output)
