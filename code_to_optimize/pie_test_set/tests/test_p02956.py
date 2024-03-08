from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02956_0():
    input_content = "3\n-1 3\n2 1\n3 -2"
    expected_output = "13"
    run_pie_test_case("../p02956.py", input_content, expected_output)


def test_problem_p02956_1():
    input_content = "3\n-1 3\n2 1\n3 -2"
    expected_output = "13"
    run_pie_test_case("../p02956.py", input_content, expected_output)


def test_problem_p02956_2():
    input_content = "10\n19 -11\n-3 -12\n5 3\n3 -15\n8 -14\n-9 -20\n10 -9\n0 2\n-7 17\n6 -6"
    expected_output = "7222"
    run_pie_test_case("../p02956.py", input_content, expected_output)


def test_problem_p02956_3():
    input_content = "4\n1 4\n2 1\n3 3\n4 2"
    expected_output = "34"
    run_pie_test_case("../p02956.py", input_content, expected_output)
