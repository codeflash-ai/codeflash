from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00072_0():
    input_content = "4\n4\n0,1,1500\n0,2,2000\n1,2,600\n1,3,500\n0"
    expected_output = "23"
    run_pie_test_case("../p00072.py", input_content, expected_output)


def test_problem_p00072_1():
    input_content = "4\n4\n0,1,1500\n0,2,2000\n1,2,600\n1,3,500\n0"
    expected_output = "23"
    run_pie_test_case("../p00072.py", input_content, expected_output)
