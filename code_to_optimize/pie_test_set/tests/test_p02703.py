from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02703_0():
    input_content = "3 2 1\n1 2 1 2\n1 3 2 4\n1 11\n1 2\n2 5"
    expected_output = "2\n14"
    run_pie_test_case("../p02703.py", input_content, expected_output)


def test_problem_p02703_1():
    input_content = "2 1 0\n1 2 1 1\n1 1000000000\n1 1"
    expected_output = "1000000001"
    run_pie_test_case("../p02703.py", input_content, expected_output)


def test_problem_p02703_2():
    input_content = "3 2 1\n1 2 1 2\n1 3 2 4\n1 11\n1 2\n2 5"
    expected_output = "2\n14"
    run_pie_test_case("../p02703.py", input_content, expected_output)


def test_problem_p02703_3():
    input_content = "4 6 1000000000\n1 2 50 1\n1 3 50 5\n1 4 50 7\n2 3 50 2\n2 4 50 4\n3 4 50 3\n10 2\n4 4\n5 5\n7 7"
    expected_output = "1\n3\n5"
    run_pie_test_case("../p02703.py", input_content, expected_output)


def test_problem_p02703_4():
    input_content = "6 5 1\n1 2 1 1\n1 3 2 1\n2 4 5 1\n3 5 11 1\n1 6 50 1\n1 10000\n1 3000\n1 700\n1 100\n1 1\n100 1"
    expected_output = "1\n9003\n14606\n16510\n16576"
    run_pie_test_case("../p02703.py", input_content, expected_output)


def test_problem_p02703_5():
    input_content = "4 4 1\n1 2 1 5\n1 3 4 4\n2 4 2 2\n3 4 1 1\n3 1\n3 1\n5 2\n6 4"
    expected_output = "5\n5\n7"
    run_pie_test_case("../p02703.py", input_content, expected_output)
