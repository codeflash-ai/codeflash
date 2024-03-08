from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03091_0():
    input_content = "7 9\n1 2\n1 3\n2 3\n1 4\n1 5\n4 5\n1 6\n1 7\n6 7"
    expected_output = "Yes"
    run_pie_test_case("../p03091.py", input_content, expected_output)


def test_problem_p03091_1():
    input_content = "3 3\n1 2\n2 3\n3 1"
    expected_output = "No"
    run_pie_test_case("../p03091.py", input_content, expected_output)


def test_problem_p03091_2():
    input_content = "18 27\n17 7\n12 15\n18 17\n13 18\n13 6\n5 7\n7 1\n14 5\n15 11\n7 6\n1 9\n5 4\n18 16\n4 6\n7 2\n7 11\n6 3\n12 14\n5 2\n10 5\n7 8\n10 15\n3 15\n9 8\n7 15\n5 16\n18 15"
    expected_output = "Yes"
    run_pie_test_case("../p03091.py", input_content, expected_output)


def test_problem_p03091_3():
    input_content = "7 9\n1 2\n1 3\n2 3\n1 4\n1 5\n4 5\n1 6\n1 7\n6 7"
    expected_output = "Yes"
    run_pie_test_case("../p03091.py", input_content, expected_output)
