from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00067_0():
    input_content = "111100001111\n111000001111\n110000001111\n100000001111\n000100010000\n000000111000\n000001111100\n100011111110\n110001111100\n111000111000\n111100010000\n000000000000\n\n010001111100\n110010000010\n010010000001\n010000000001\n010000000110\n010000111000\n010000000100\n010000000010\n010000000001\n010010000001\n010010000010\n111001111100\n\n000000000000\n111111111111\n100010100001\n100010100001\n100010100001\n100010100001\n100100100101\n101000011101\n100000000001\n100000000001\n111111111111\n100000000001"
    expected_output = "5\n13\n4"
    run_pie_test_case("../p00067.py", input_content, expected_output)


def test_problem_p00067_1():
    input_content = "111100001111\n111000001111\n110000001111\n100000001111\n000100010000\n000000111000\n000001111100\n100011111110\n110001111100\n111000111000\n111100010000\n000000000000\n\n010001111100\n110010000010\n010010000001\n010000000001\n010000000110\n010000111000\n010000000100\n010000000010\n010000000001\n010010000001\n010010000010\n111001111100\n\n000000000000\n111111111111\n100010100001\n100010100001\n100010100001\n100010100001\n100100100101\n101000011101\n100000000001\n100000000001\n111111111111\n100000000001"
    expected_output = "5\n13\n4"
    run_pie_test_case("../p00067.py", input_content, expected_output)
