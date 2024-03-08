from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02925_0():
    input_content = "3\n2 3\n1 3\n1 2"
    expected_output = "3"
    run_pie_test_case("../p02925.py", input_content, expected_output)


def test_problem_p02925_1():
    input_content = "4\n2 3 4\n1 3 4\n4 1 2\n3 1 2"
    expected_output = "4"
    run_pie_test_case("../p02925.py", input_content, expected_output)


def test_problem_p02925_2():
    input_content = "3\n2 3\n1 3\n1 2"
    expected_output = "3"
    run_pie_test_case("../p02925.py", input_content, expected_output)


def test_problem_p02925_3():
    input_content = "3\n2 3\n3 1\n1 2"
    expected_output = "-1"
    run_pie_test_case("../p02925.py", input_content, expected_output)
