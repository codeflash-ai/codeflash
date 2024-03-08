from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02573_0():
    input_content = "5 3\n1 2\n3 4\n5 1"
    expected_output = "3"
    run_pie_test_case("../p02573.py", input_content, expected_output)


def test_problem_p02573_1():
    input_content = "10 4\n3 1\n4 1\n5 9\n2 6"
    expected_output = "3"
    run_pie_test_case("../p02573.py", input_content, expected_output)


def test_problem_p02573_2():
    input_content = "5 3\n1 2\n3 4\n5 1"
    expected_output = "3"
    run_pie_test_case("../p02573.py", input_content, expected_output)


def test_problem_p02573_3():
    input_content = "4 10\n1 2\n2 1\n1 2\n2 1\n1 2\n1 3\n1 4\n2 3\n2 4\n3 4"
    expected_output = "4"
    run_pie_test_case("../p02573.py", input_content, expected_output)
