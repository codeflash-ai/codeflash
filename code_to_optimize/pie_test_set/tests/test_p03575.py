from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03575_0():
    input_content = "7 7\n1 3\n2 7\n3 4\n4 5\n4 6\n5 6\n6 7"
    expected_output = "4"
    run_pie_test_case("../p03575.py", input_content, expected_output)


def test_problem_p03575_1():
    input_content = "3 3\n1 2\n1 3\n2 3"
    expected_output = "0"
    run_pie_test_case("../p03575.py", input_content, expected_output)


def test_problem_p03575_2():
    input_content = "6 5\n1 2\n2 3\n3 4\n4 5\n5 6"
    expected_output = "5"
    run_pie_test_case("../p03575.py", input_content, expected_output)


def test_problem_p03575_3():
    input_content = "7 7\n1 3\n2 7\n3 4\n4 5\n4 6\n5 6\n6 7"
    expected_output = "4"
    run_pie_test_case("../p03575.py", input_content, expected_output)
