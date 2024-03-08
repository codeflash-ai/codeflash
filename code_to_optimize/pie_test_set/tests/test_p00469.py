from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00469_0():
    input_content = "4\n2\n1\n2\n12\n1\n6\n3\n72\n2\n12\n7\n2\n1\n0\n0"
    expected_output = "7\n68"
    run_pie_test_case("../p00469.py", input_content, expected_output)


def test_problem_p00469_1():
    input_content = "4\n2\n1\n2\n12\n1\n6\n3\n72\n2\n12\n7\n2\n1\n0\n0"
    expected_output = "7\n68"
    run_pie_test_case("../p00469.py", input_content, expected_output)
