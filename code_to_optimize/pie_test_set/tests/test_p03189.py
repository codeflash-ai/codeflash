from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03189_0():
    input_content = "3 2\n1\n2\n3\n1 2\n1 3"
    expected_output = "6"
    run_pie_test_case("../p03189.py", input_content, expected_output)


def test_problem_p03189_1():
    input_content = "3 2\n1\n2\n3\n1 2\n1 3"
    expected_output = "6"
    run_pie_test_case("../p03189.py", input_content, expected_output)


def test_problem_p03189_2():
    input_content = "5 3\n3\n2\n3\n1\n4\n1 5\n2 3\n4 2"
    expected_output = "36"
    run_pie_test_case("../p03189.py", input_content, expected_output)


def test_problem_p03189_3():
    input_content = "9 5\n3\n1\n4\n1\n5\n9\n2\n6\n5\n3 5\n8 9\n7 9\n3 2\n3 8"
    expected_output = "425"
    run_pie_test_case("../p03189.py", input_content, expected_output)
