from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03768_0():
    input_content = "7 7\n1 2\n1 3\n1 4\n4 5\n5 6\n5 7\n2 3\n2\n6 1 1\n1 2 2"
    expected_output = "2\n2\n2\n2\n2\n1\n0"
    run_pie_test_case("../p03768.py", input_content, expected_output)


def test_problem_p03768_1():
    input_content = "14 10\n1 4\n5 7\n7 11\n4 10\n14 7\n14 3\n6 14\n8 11\n5 13\n8 3\n8\n8 6 2\n9 7 85\n6 9 3\n6 7 5\n10 3 1\n12 9 4\n9 6 6\n8 2 3"
    expected_output = "1\n0\n3\n1\n5\n5\n3\n3\n6\n1\n3\n4\n5\n3"
    run_pie_test_case("../p03768.py", input_content, expected_output)


def test_problem_p03768_2():
    input_content = "7 7\n1 2\n1 3\n1 4\n4 5\n5 6\n5 7\n2 3\n2\n6 1 1\n1 2 2"
    expected_output = "2\n2\n2\n2\n2\n1\n0"
    run_pie_test_case("../p03768.py", input_content, expected_output)
