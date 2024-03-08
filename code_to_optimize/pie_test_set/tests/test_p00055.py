from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00055_0():
    input_content = "1.0\n2.0\n3.0"
    expected_output = "7.81481481\n15.62962963\n23.44444444"
    run_pie_test_case("../p00055.py", input_content, expected_output)


def test_problem_p00055_1():
    input_content = "1.0\n2.0\n3.0"
    expected_output = "7.81481481\n15.62962963\n23.44444444"
    run_pie_test_case("../p00055.py", input_content, expected_output)
