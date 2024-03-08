from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02998_0():
    input_content = "3\n1 1\n5 1\n5 5"
    expected_output = "1"
    run_pie_test_case("../p02998.py", input_content, expected_output)


def test_problem_p02998_1():
    input_content = "3\n1 1\n5 1\n5 5"
    expected_output = "1"
    run_pie_test_case("../p02998.py", input_content, expected_output)


def test_problem_p02998_2():
    input_content = "9\n1 1\n2 1\n3 1\n4 1\n5 1\n1 2\n1 3\n1 4\n1 5"
    expected_output = "16"
    run_pie_test_case("../p02998.py", input_content, expected_output)


def test_problem_p02998_3():
    input_content = "2\n10 10\n20 20"
    expected_output = "0"
    run_pie_test_case("../p02998.py", input_content, expected_output)
