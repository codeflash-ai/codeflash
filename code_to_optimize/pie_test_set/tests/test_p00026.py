from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00026_0():
    input_content = "2,5,3\n3,6,1\n3,4,2\n4,5,2\n3,6,3\n2,4,1"
    expected_output = "77\n5"
    run_pie_test_case("../p00026.py", input_content, expected_output)


def test_problem_p00026_1():
    input_content = "2,5,3\n3,6,1\n3,4,2\n4,5,2\n3,6,3\n2,4,1"
    expected_output = "77\n5"
    run_pie_test_case("../p00026.py", input_content, expected_output)
