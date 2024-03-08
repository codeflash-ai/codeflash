from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03647_0():
    input_content = "3 2\n1 2\n2 3"
    expected_output = "POSSIBLE"
    run_pie_test_case("../p03647.py", input_content, expected_output)


def test_problem_p03647_1():
    input_content = "5 5\n1 3\n4 5\n2 3\n2 4\n1 4"
    expected_output = "POSSIBLE"
    run_pie_test_case("../p03647.py", input_content, expected_output)


def test_problem_p03647_2():
    input_content = "100000 1\n1 99999"
    expected_output = "IMPOSSIBLE"
    run_pie_test_case("../p03647.py", input_content, expected_output)


def test_problem_p03647_3():
    input_content = "4 3\n1 2\n2 3\n3 4"
    expected_output = "IMPOSSIBLE"
    run_pie_test_case("../p03647.py", input_content, expected_output)


def test_problem_p03647_4():
    input_content = "3 2\n1 2\n2 3"
    expected_output = "POSSIBLE"
    run_pie_test_case("../p03647.py", input_content, expected_output)
