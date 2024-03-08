from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00534_0():
    input_content = "3 5\n10\n25\n15\n50\n30\n15\n40\n30"
    expected_output = "1125"
    run_pie_test_case("../p00534.py", input_content, expected_output)


def test_problem_p00534_1():
    input_content = "3 5\n10\n25\n15\n50\n30\n15\n40\n30"
    expected_output = "1125"
    run_pie_test_case("../p00534.py", input_content, expected_output)
