from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04028_0():
    input_content = "3\n0"
    expected_output = "5"
    run_pie_test_case("../p04028.py", input_content, expected_output)


def test_problem_p04028_1():
    input_content = "5000\n01000001011101000100001101101111011001000110010101110010000"
    expected_output = "500886057"
    run_pie_test_case("../p04028.py", input_content, expected_output)


def test_problem_p04028_2():
    input_content = "3\n0"
    expected_output = "5"
    run_pie_test_case("../p04028.py", input_content, expected_output)


def test_problem_p04028_3():
    input_content = "300\n1100100"
    expected_output = "519054663"
    run_pie_test_case("../p04028.py", input_content, expected_output)
