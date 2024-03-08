from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02936_0():
    input_content = "4 3\n1 2\n2 3\n2 4\n2 10\n1 100\n3 1"
    expected_output = "100 110 111 110"
    run_pie_test_case("../p02936.py", input_content, expected_output)


def test_problem_p02936_1():
    input_content = "6 2\n1 2\n1 3\n2 4\n3 6\n2 5\n1 10\n1 10"
    expected_output = "20 20 20 20 20 20"
    run_pie_test_case("../p02936.py", input_content, expected_output)


def test_problem_p02936_2():
    input_content = "4 3\n1 2\n2 3\n2 4\n2 10\n1 100\n3 1"
    expected_output = "100 110 111 110"
    run_pie_test_case("../p02936.py", input_content, expected_output)
