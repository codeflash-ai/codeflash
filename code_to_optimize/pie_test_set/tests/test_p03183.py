from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03183_0():
    input_content = "3\n2 2 20\n2 1 30\n3 1 40"
    expected_output = "50"
    run_pie_test_case("../p03183.py", input_content, expected_output)


def test_problem_p03183_1():
    input_content = "5\n1 10000 1000000000\n1 10000 1000000000\n1 10000 1000000000\n1 10000 1000000000\n1 10000 1000000000"
    expected_output = "5000000000"
    run_pie_test_case("../p03183.py", input_content, expected_output)


def test_problem_p03183_2():
    input_content = "3\n2 2 20\n2 1 30\n3 1 40"
    expected_output = "50"
    run_pie_test_case("../p03183.py", input_content, expected_output)


def test_problem_p03183_3():
    input_content = "4\n1 2 10\n3 1 10\n2 4 10\n1 6 10"
    expected_output = "40"
    run_pie_test_case("../p03183.py", input_content, expected_output)


def test_problem_p03183_4():
    input_content = "8\n9 5 7\n6 2 7\n5 7 3\n7 8 8\n1 9 6\n3 3 3\n4 1 7\n4 5 5"
    expected_output = "22"
    run_pie_test_case("../p03183.py", input_content, expected_output)
