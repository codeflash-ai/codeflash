from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03181_0():
    input_content = "3 100\n1 2\n2 3"
    expected_output = "3\n4\n3"
    run_pie_test_case("../p03181.py", input_content, expected_output)


def test_problem_p03181_1():
    input_content = "3 100\n1 2\n2 3"
    expected_output = "3\n4\n3"
    run_pie_test_case("../p03181.py", input_content, expected_output)


def test_problem_p03181_2():
    input_content = "1 100"
    expected_output = "1"
    run_pie_test_case("../p03181.py", input_content, expected_output)


def test_problem_p03181_3():
    input_content = "10 2\n8 5\n10 8\n6 5\n1 5\n4 8\n2 10\n3 6\n9 2\n1 7"
    expected_output = "0\n0\n1\n1\n1\n0\n1\n0\n1\n1"
    run_pie_test_case("../p03181.py", input_content, expected_output)


def test_problem_p03181_4():
    input_content = "4 100\n1 2\n1 3\n1 4"
    expected_output = "8\n5\n5\n5"
    run_pie_test_case("../p03181.py", input_content, expected_output)
