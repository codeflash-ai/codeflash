from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04000_0():
    input_content = "4 5 8\n1 1\n1 4\n1 5\n2 3\n3 1\n3 2\n3 4\n4 4"
    expected_output = "0\n0\n0\n2\n4\n0\n0\n0\n0\n0"
    run_pie_test_case("../p04000.py", input_content, expected_output)


def test_problem_p04000_1():
    input_content = "1000000000 1000000000 0"
    expected_output = "999999996000000004\n0\n0\n0\n0\n0\n0\n0\n0\n0"
    run_pie_test_case("../p04000.py", input_content, expected_output)


def test_problem_p04000_2():
    input_content = "4 5 8\n1 1\n1 4\n1 5\n2 3\n3 1\n3 2\n3 4\n4 4"
    expected_output = "0\n0\n0\n2\n4\n0\n0\n0\n0\n0"
    run_pie_test_case("../p04000.py", input_content, expected_output)


def test_problem_p04000_3():
    input_content = "10 10 20\n1 1\n1 4\n1 9\n2 5\n3 10\n4 2\n4 7\n5 9\n6 4\n6 6\n6 7\n7 1\n7 3\n7 7\n8 1\n8 5\n8 10\n9 2\n10 4\n10 9"
    expected_output = "4\n26\n22\n10\n2\n0\n0\n0\n0\n0"
    run_pie_test_case("../p04000.py", input_content, expected_output)
