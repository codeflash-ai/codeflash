from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03964_0():
    input_content = "3\n2 3\n1 1\n3 2"
    expected_output = "10"
    run_pie_test_case("../p03964.py", input_content, expected_output)


def test_problem_p03964_1():
    input_content = "5\n3 10\n48 17\n31 199\n231 23\n3 2"
    expected_output = "6930"
    run_pie_test_case("../p03964.py", input_content, expected_output)


def test_problem_p03964_2():
    input_content = "4\n1 1\n1 1\n1 5\n1 100"
    expected_output = "101"
    run_pie_test_case("../p03964.py", input_content, expected_output)


def test_problem_p03964_3():
    input_content = "3\n2 3\n1 1\n3 2"
    expected_output = "10"
    run_pie_test_case("../p03964.py", input_content, expected_output)
