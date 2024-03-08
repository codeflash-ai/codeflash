from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00134_0():
    input_content = "6\n12300\n5600\n33800\n0\n26495\n52000"
    expected_output = "21699"
    run_pie_test_case("../p00134.py", input_content, expected_output)


def test_problem_p00134_1():
    input_content = "6\n12300\n5600\n33800\n0\n26495\n52000"
    expected_output = "21699"
    run_pie_test_case("../p00134.py", input_content, expected_output)
