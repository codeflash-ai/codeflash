from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02928_0():
    input_content = "2 2\n2 1"
    expected_output = "3"
    run_pie_test_case("../p02928.py", input_content, expected_output)


def test_problem_p02928_1():
    input_content = "2 2\n2 1"
    expected_output = "3"
    run_pie_test_case("../p02928.py", input_content, expected_output)


def test_problem_p02928_2():
    input_content = "10 998244353\n10 9 8 7 5 6 3 4 2 1"
    expected_output = "185297239"
    run_pie_test_case("../p02928.py", input_content, expected_output)


def test_problem_p02928_3():
    input_content = "3 5\n1 1 1"
    expected_output = "0"
    run_pie_test_case("../p02928.py", input_content, expected_output)
