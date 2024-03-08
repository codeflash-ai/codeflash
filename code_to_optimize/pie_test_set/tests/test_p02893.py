from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02893_0():
    input_content = "3\n111"
    expected_output = "40"
    run_pie_test_case("../p02893.py", input_content, expected_output)


def test_problem_p02893_1():
    input_content = "30\n001110011011011101010111011100"
    expected_output = "549320998"
    run_pie_test_case("../p02893.py", input_content, expected_output)


def test_problem_p02893_2():
    input_content = "3\n111"
    expected_output = "40"
    run_pie_test_case("../p02893.py", input_content, expected_output)


def test_problem_p02893_3():
    input_content = "6\n110101"
    expected_output = "616"
    run_pie_test_case("../p02893.py", input_content, expected_output)
