from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03240_0():
    input_content = "4\n2 3 5\n2 1 5\n1 2 5\n3 2 5"
    expected_output = "2 2 6"
    run_pie_test_case("../p03240.py", input_content, expected_output)


def test_problem_p03240_1():
    input_content = "2\n0 0 100\n1 1 98"
    expected_output = "0 0 100"
    run_pie_test_case("../p03240.py", input_content, expected_output)


def test_problem_p03240_2():
    input_content = "3\n99 1 191\n100 1 192\n99 0 192"
    expected_output = "100 0 193"
    run_pie_test_case("../p03240.py", input_content, expected_output)


def test_problem_p03240_3():
    input_content = "4\n2 3 5\n2 1 5\n1 2 5\n3 2 5"
    expected_output = "2 2 6"
    run_pie_test_case("../p03240.py", input_content, expected_output)
