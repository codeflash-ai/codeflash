from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02868_0():
    input_content = "4 3\n1 3 2\n2 4 3\n1 4 6"
    expected_output = "5"
    run_pie_test_case("../p02868.py", input_content, expected_output)


def test_problem_p02868_1():
    input_content = "4 3\n1 3 2\n2 4 3\n1 4 6"
    expected_output = "5"
    run_pie_test_case("../p02868.py", input_content, expected_output)


def test_problem_p02868_2():
    input_content = "10 7\n1 5 18\n3 4 8\n1 3 5\n4 7 10\n5 9 8\n6 10 5\n8 10 3"
    expected_output = "28"
    run_pie_test_case("../p02868.py", input_content, expected_output)


def test_problem_p02868_3():
    input_content = "4 2\n1 2 1\n3 4 2"
    expected_output = "-1"
    run_pie_test_case("../p02868.py", input_content, expected_output)
