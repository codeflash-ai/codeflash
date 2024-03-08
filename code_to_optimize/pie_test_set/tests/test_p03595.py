from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03595_0():
    input_content = "2 2\n10\n01\n10\n01"
    expected_output = "6"
    run_pie_test_case("../p03595.py", input_content, expected_output)


def test_problem_p03595_1():
    input_content = "3 4\n111\n111\n1111\n1111"
    expected_output = "1276"
    run_pie_test_case("../p03595.py", input_content, expected_output)


def test_problem_p03595_2():
    input_content = "9 13\n111100001\n010101011\n0000000000000\n1010111111101"
    expected_output = "177856"
    run_pie_test_case("../p03595.py", input_content, expected_output)


def test_problem_p03595_3():
    input_content = "2 2\n11\n11\n11\n11"
    expected_output = "32"
    run_pie_test_case("../p03595.py", input_content, expected_output)


def test_problem_p03595_4():
    input_content = (
        "17 21\n11001010101011101\n11001010011010111\n111010101110101111100\n011010110110101000111"
    )
    expected_output = "548356548"
    run_pie_test_case("../p03595.py", input_content, expected_output)


def test_problem_p03595_5():
    input_content = "23 30\n01010010101010010001110\n11010100100100101010101\n000101001001010010101010101101\n101001000100101001010010101000"
    expected_output = "734524988"
    run_pie_test_case("../p03595.py", input_content, expected_output)


def test_problem_p03595_6():
    input_content = "2 2\n10\n01\n10\n01"
    expected_output = "6"
    run_pie_test_case("../p03595.py", input_content, expected_output)


def test_problem_p03595_7():
    input_content = "3 4\n000\n101\n1111\n0010"
    expected_output = "21"
    run_pie_test_case("../p03595.py", input_content, expected_output)
