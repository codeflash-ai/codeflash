from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03975_0():
    input_content = "5 5 9\n4\n3\n6\n9\n1"
    expected_output = "4"
    run_pie_test_case("../p03975.py", input_content, expected_output)


def test_problem_p03975_1():
    input_content = "4 3 6\n9\n6\n8\n1"
    expected_output = "4"
    run_pie_test_case("../p03975.py", input_content, expected_output)


def test_problem_p03975_2():
    input_content = "5 5 9\n4\n3\n6\n9\n1"
    expected_output = "4"
    run_pie_test_case("../p03975.py", input_content, expected_output)


def test_problem_p03975_3():
    input_content = "2 1 2\n1\n2"
    expected_output = "1"
    run_pie_test_case("../p03975.py", input_content, expected_output)


def test_problem_p03975_4():
    input_content = "5 4 9\n5\n6\n7\n8\n9"
    expected_output = "1"
    run_pie_test_case("../p03975.py", input_content, expected_output)
