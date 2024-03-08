from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00181_0():
    input_content = "3 9\n500\n300\n800\n200\n100\n600\n900\n700\n400\n4 3\n1000\n1000\n1000\n0 0"
    expected_output = "1800\n1000"
    run_pie_test_case("../p00181.py", input_content, expected_output)


def test_problem_p00181_1():
    input_content = "3 9\n500\n300\n800\n200\n100\n600\n900\n700\n400\n4 3\n1000\n1000\n1000\n0 0"
    expected_output = "1800\n1000"
    run_pie_test_case("../p00181.py", input_content, expected_output)
