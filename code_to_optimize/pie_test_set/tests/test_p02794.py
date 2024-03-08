from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02794_0():
    input_content = "3\n1 2\n2 3\n1\n1 3"
    expected_output = "3"
    run_pie_test_case("../p02794.py", input_content, expected_output)


def test_problem_p02794_1():
    input_content = "2\n1 2\n1\n1 2"
    expected_output = "1"
    run_pie_test_case("../p02794.py", input_content, expected_output)


def test_problem_p02794_2():
    input_content = "5\n1 2\n3 2\n3 4\n5 3\n3\n1 3\n2 4\n2 5"
    expected_output = "9"
    run_pie_test_case("../p02794.py", input_content, expected_output)


def test_problem_p02794_3():
    input_content = "3\n1 2\n2 3\n1\n1 3"
    expected_output = "3"
    run_pie_test_case("../p02794.py", input_content, expected_output)


def test_problem_p02794_4():
    input_content = "8\n1 2\n2 3\n4 3\n2 5\n6 3\n6 7\n8 6\n5\n2 7\n3 5\n1 6\n2 8\n7 8"
    expected_output = "62"
    run_pie_test_case("../p02794.py", input_content, expected_output)
