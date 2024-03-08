from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00825_0():
    input_content = "4\n1 2 10\n2 3 10\n3 3 10\n1 3 10\n6\n1 20 1000\n3 25 10000\n5 15 5000\n22 300 5500\n10 295 9000\n7 7 6000\n8\n32 251 2261\n123 281 1339\n211 235 5641\n162 217 7273\n22 139 7851\n194 198 9190\n119 274 878\n122 173 8640\n0"
    expected_output = "30\n25500\n38595"
    run_pie_test_case("../p00825.py", input_content, expected_output)


def test_problem_p00825_1():
    input_content = "4\n1 2 10\n2 3 10\n3 3 10\n1 3 10\n6\n1 20 1000\n3 25 10000\n5 15 5000\n22 300 5500\n10 295 9000\n7 7 6000\n8\n32 251 2261\n123 281 1339\n211 235 5641\n162 217 7273\n22 139 7851\n194 198 9190\n119 274 878\n122 173 8640\n0"
    expected_output = "30\n25500\n38595"
    run_pie_test_case("../p00825.py", input_content, expected_output)
