from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02386_0():
    input_content = "3\n1 2 3 4 5 6\n6 2 4 3 5 1\n6 5 4 3 2 1"
    expected_output = "No"
    run_pie_test_case("../p02386.py", input_content, expected_output)


def test_problem_p02386_1():
    input_content = "3\n1 2 3 4 5 6\n6 5 4 3 2 1\n5 4 3 2 1 6"
    expected_output = "Yes"
    run_pie_test_case("../p02386.py", input_content, expected_output)


def test_problem_p02386_2():
    input_content = "3\n1 2 3 4 5 6\n6 2 4 3 5 1\n6 5 4 3 2 1"
    expected_output = "No"
    run_pie_test_case("../p02386.py", input_content, expected_output)
