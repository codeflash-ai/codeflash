from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03426_0():
    input_content = "3 3 2\n1 4 3\n2 5 7\n8 9 6\n1\n4 8"
    expected_output = "5"
    run_pie_test_case("../p03426.py", input_content, expected_output)


def test_problem_p03426_1():
    input_content = "3 3 2\n1 4 3\n2 5 7\n8 9 6\n1\n4 8"
    expected_output = "5"
    run_pie_test_case("../p03426.py", input_content, expected_output)


def test_problem_p03426_2():
    input_content = "4 2 3\n3 7\n1 4\n5 2\n6 8\n2\n2 2\n2 2"
    expected_output = "0\n0"
    run_pie_test_case("../p03426.py", input_content, expected_output)


def test_problem_p03426_3():
    input_content = "5 5 4\n13 25 7 15 17\n16 22 20 2 9\n14 11 12 1 19\n10 6 23 8 18\n3 21 5 24 4\n3\n13 13\n2 10\n13 13"
    expected_output = "0\n5\n0"
    run_pie_test_case("../p03426.py", input_content, expected_output)
