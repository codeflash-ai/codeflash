from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02834_0():
    input_content = "5 4 1\n1 2\n2 3\n3 4\n3 5"
    expected_output = "2"
    run_pie_test_case("../p02834.py", input_content, expected_output)


def test_problem_p02834_1():
    input_content = "2 1 2\n1 2"
    expected_output = "0"
    run_pie_test_case("../p02834.py", input_content, expected_output)


def test_problem_p02834_2():
    input_content = "5 4 5\n1 2\n1 3\n1 4\n1 5"
    expected_output = "1"
    run_pie_test_case("../p02834.py", input_content, expected_output)


def test_problem_p02834_3():
    input_content = "5 4 1\n1 2\n2 3\n3 4\n3 5"
    expected_output = "2"
    run_pie_test_case("../p02834.py", input_content, expected_output)


def test_problem_p02834_4():
    input_content = "9 6 1\n1 2\n2 3\n3 4\n4 5\n5 6\n4 7\n7 8\n8 9"
    expected_output = "5"
    run_pie_test_case("../p02834.py", input_content, expected_output)
