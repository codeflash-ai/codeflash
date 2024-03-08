from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04014_0():
    input_content = "87654\n30"
    expected_output = "10"
    run_pie_test_case("../p04014.py", input_content, expected_output)


def test_problem_p04014_1():
    input_content = "87654\n30"
    expected_output = "10"
    run_pie_test_case("../p04014.py", input_content, expected_output)


def test_problem_p04014_2():
    input_content = "87654\n138"
    expected_output = "100"
    run_pie_test_case("../p04014.py", input_content, expected_output)


def test_problem_p04014_3():
    input_content = "87654\n45678"
    expected_output = "-1"
    run_pie_test_case("../p04014.py", input_content, expected_output)


def test_problem_p04014_4():
    input_content = "31415926535\n1"
    expected_output = "31415926535"
    run_pie_test_case("../p04014.py", input_content, expected_output)


def test_problem_p04014_5():
    input_content = "1\n31415926535"
    expected_output = "-1"
    run_pie_test_case("../p04014.py", input_content, expected_output)
