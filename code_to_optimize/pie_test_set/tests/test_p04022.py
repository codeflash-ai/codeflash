from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04022_0():
    input_content = "8\n1\n2\n3\n4\n5\n6\n7\n8"
    expected_output = "6"
    run_pie_test_case("../p04022.py", input_content, expected_output)


def test_problem_p04022_1():
    input_content = "10\n1\n10\n100\n1000000007\n10000000000\n1000000009\n999999999\n999\n999\n999"
    expected_output = "9"
    run_pie_test_case("../p04022.py", input_content, expected_output)


def test_problem_p04022_2():
    input_content = "6\n2\n4\n8\n16\n32\n64"
    expected_output = "3"
    run_pie_test_case("../p04022.py", input_content, expected_output)


def test_problem_p04022_3():
    input_content = "8\n1\n2\n3\n4\n5\n6\n7\n8"
    expected_output = "6"
    run_pie_test_case("../p04022.py", input_content, expected_output)
