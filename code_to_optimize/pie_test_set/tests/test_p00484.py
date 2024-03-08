from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00484_0():
    input_content = "7 4\n14 1\n13 2\n12 3\n14 2\n8 2\n16 3\n11 2"
    expected_output = "60"
    run_pie_test_case("../p00484.py", input_content, expected_output)


def test_problem_p00484_1():
    input_content = "7 4\n14 1\n13 2\n12 3\n14 2\n8 2\n16 3\n11 2"
    expected_output = "60"
    run_pie_test_case("../p00484.py", input_content, expected_output)
