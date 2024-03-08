from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02902_0():
    input_content = "4 5\n1 2\n2 3\n2 4\n4 1\n4 3"
    expected_output = "3\n1\n2\n4"
    run_pie_test_case("../p02902.py", input_content, expected_output)


def test_problem_p02902_1():
    input_content = "6 9\n1 2\n2 3\n3 4\n4 5\n5 6\n5 1\n5 2\n6 1\n6 2"
    expected_output = "4\n2\n3\n4\n5"
    run_pie_test_case("../p02902.py", input_content, expected_output)


def test_problem_p02902_2():
    input_content = "4 5\n1 2\n2 3\n2 4\n4 1\n4 3"
    expected_output = "3\n1\n2\n4"
    run_pie_test_case("../p02902.py", input_content, expected_output)


def test_problem_p02902_3():
    input_content = "4 5\n1 2\n2 3\n2 4\n1 4\n4 3"
    expected_output = "-1"
    run_pie_test_case("../p02902.py", input_content, expected_output)
