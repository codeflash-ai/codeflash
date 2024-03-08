from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03526_0():
    input_content = "3\n0 2\n1 3\n3 4"
    expected_output = "2"
    run_pie_test_case("../p03526.py", input_content, expected_output)


def test_problem_p03526_1():
    input_content = "3\n2 4\n3 1\n4 1"
    expected_output = "3"
    run_pie_test_case("../p03526.py", input_content, expected_output)


def test_problem_p03526_2():
    input_content = "10\n1 3\n8 4\n8 3\n9 1\n6 4\n2 3\n4 2\n9 2\n8 3\n0 1"
    expected_output = "5"
    run_pie_test_case("../p03526.py", input_content, expected_output)


def test_problem_p03526_3():
    input_content = "3\n0 2\n1 3\n3 4"
    expected_output = "2"
    run_pie_test_case("../p03526.py", input_content, expected_output)
