from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02867_0():
    input_content = "3\n1 3 2\n1 2 3"
    expected_output = "Yes"
    run_pie_test_case("../p02867.py", input_content, expected_output)


def test_problem_p02867_1():
    input_content = "3\n1 2 3\n2 2 2"
    expected_output = "No"
    run_pie_test_case("../p02867.py", input_content, expected_output)


def test_problem_p02867_2():
    input_content = "6\n3 1 2 6 3 4\n2 2 8 3 4 3"
    expected_output = "Yes"
    run_pie_test_case("../p02867.py", input_content, expected_output)


def test_problem_p02867_3():
    input_content = "3\n1 3 2\n1 2 3"
    expected_output = "Yes"
    run_pie_test_case("../p02867.py", input_content, expected_output)
