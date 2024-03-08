from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02580_0():
    input_content = "2 3 3\n2 2\n1 1\n1 3"
    expected_output = "3"
    run_pie_test_case("../p02580.py", input_content, expected_output)


def test_problem_p02580_1():
    input_content = "2 3 3\n2 2\n1 1\n1 3"
    expected_output = "3"
    run_pie_test_case("../p02580.py", input_content, expected_output)


def test_problem_p02580_2():
    input_content = "3 3 4\n3 3\n3 1\n1 1\n1 2"
    expected_output = "3"
    run_pie_test_case("../p02580.py", input_content, expected_output)


def test_problem_p02580_3():
    input_content = "5 5 10\n2 5\n4 3\n2 3\n5 5\n2 2\n5 4\n5 3\n5 1\n3 5\n1 4"
    expected_output = "6"
    run_pie_test_case("../p02580.py", input_content, expected_output)
