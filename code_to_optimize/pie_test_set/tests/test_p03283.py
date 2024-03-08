from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03283_0():
    input_content = "2 3 1\n1 1\n1 2\n2 2\n1 2"
    expected_output = "3"
    run_pie_test_case("../p03283.py", input_content, expected_output)


def test_problem_p03283_1():
    input_content = "10 3 2\n1 5\n2 8\n7 10\n1 7\n3 10"
    expected_output = "1\n1"
    run_pie_test_case("../p03283.py", input_content, expected_output)


def test_problem_p03283_2():
    input_content = "2 3 1\n1 1\n1 2\n2 2\n1 2"
    expected_output = "3"
    run_pie_test_case("../p03283.py", input_content, expected_output)


def test_problem_p03283_3():
    input_content = "10 10 10\n1 6\n2 9\n4 5\n4 7\n4 7\n5 8\n6 6\n6 7\n7 9\n10 10\n1 8\n1 9\n1 10\n2 8\n2 9\n2 10\n3 8\n3 9\n3 10\n1 10"
    expected_output = "7\n9\n10\n6\n8\n9\n6\n7\n8\n10"
    run_pie_test_case("../p03283.py", input_content, expected_output)
