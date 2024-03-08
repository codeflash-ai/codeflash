from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02779_0():
    input_content = "5\n2 6 1 4 5"
    expected_output = "YES"
    run_pie_test_case("../p02779.py", input_content, expected_output)


def test_problem_p02779_1():
    input_content = "6\n4 1 3 1 6 2"
    expected_output = "NO"
    run_pie_test_case("../p02779.py", input_content, expected_output)


def test_problem_p02779_2():
    input_content = "2\n10000000 10000000"
    expected_output = "NO"
    run_pie_test_case("../p02779.py", input_content, expected_output)


def test_problem_p02779_3():
    input_content = "5\n2 6 1 4 5"
    expected_output = "YES"
    run_pie_test_case("../p02779.py", input_content, expected_output)
