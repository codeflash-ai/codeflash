from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04027_0():
    input_content = "2 3\n1 1\n1 1"
    expected_output = "4"
    run_pie_test_case("../p04027.py", input_content, expected_output)


def test_problem_p04027_1():
    input_content = "3 100\n7 6 5\n9 9 9"
    expected_output = "139123417"
    run_pie_test_case("../p04027.py", input_content, expected_output)


def test_problem_p04027_2():
    input_content = "1 2\n1\n3"
    expected_output = "14"
    run_pie_test_case("../p04027.py", input_content, expected_output)


def test_problem_p04027_3():
    input_content = "2 3\n1 1\n1 1"
    expected_output = "4"
    run_pie_test_case("../p04027.py", input_content, expected_output)


def test_problem_p04027_4():
    input_content = "2 3\n1 1\n2 2"
    expected_output = "66"
    run_pie_test_case("../p04027.py", input_content, expected_output)


def test_problem_p04027_5():
    input_content = "4 8\n3 1 4 1\n3 1 4 1"
    expected_output = "421749"
    run_pie_test_case("../p04027.py", input_content, expected_output)
