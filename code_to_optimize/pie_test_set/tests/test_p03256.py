from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03256_0():
    input_content = "2 3\nAB\n1 1\n1 2\n2 2"
    expected_output = "Yes"
    run_pie_test_case("../p03256.py", input_content, expected_output)


def test_problem_p03256_1():
    input_content = "13 17\nBBABBBAABABBA\n7 1\n7 9\n11 12\n3 9\n11 9\n2 1\n11 5\n12 11\n10 8\n1 11\n1 8\n7 7\n9 10\n8 8\n8 12\n6 2\n13 11"
    expected_output = "No"
    run_pie_test_case("../p03256.py", input_content, expected_output)


def test_problem_p03256_2():
    input_content = "2 3\nAB\n1 1\n1 2\n2 2"
    expected_output = "Yes"
    run_pie_test_case("../p03256.py", input_content, expected_output)


def test_problem_p03256_3():
    input_content = "4 3\nABAB\n1 2\n2 3\n3 1"
    expected_output = "No"
    run_pie_test_case("../p03256.py", input_content, expected_output)


def test_problem_p03256_4():
    input_content = "13 23\nABAAAABBBBAAB\n7 1\n10 6\n1 11\n2 10\n2 8\n2 11\n11 12\n8 3\n7 12\n11 2\n13 13\n11 9\n4 1\n9 7\n9 6\n8 13\n8 6\n4 10\n8 7\n4 3\n2 1\n8 12\n6 9"
    expected_output = "Yes"
    run_pie_test_case("../p03256.py", input_content, expected_output)
