from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00069_0():
    input_content = "5\n2\n3\n9\n1010\n1001\n0100\n1001\n0010\n1000\n0100\n0101\n1010\n0"
    expected_output = "6 4"
    run_pie_test_case("../p00069.py", input_content, expected_output)


def test_problem_p00069_1():
    input_content = "5\n2\n3\n9\n1010\n1001\n0100\n1001\n0010\n1000\n0100\n0101\n1010\n0"
    expected_output = "6 4"
    run_pie_test_case("../p00069.py", input_content, expected_output)
