from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03171_0():
    input_content = "4\n10 80 90 30"
    expected_output = "10"
    run_pie_test_case("../p03171.py", input_content, expected_output)


def test_problem_p03171_1():
    input_content = "4\n10 80 90 30"
    expected_output = "10"
    run_pie_test_case("../p03171.py", input_content, expected_output)


def test_problem_p03171_2():
    input_content = "10\n1000000000 1 1000000000 1 1000000000 1 1000000000 1 1000000000 1"
    expected_output = "4999999995"
    run_pie_test_case("../p03171.py", input_content, expected_output)


def test_problem_p03171_3():
    input_content = "3\n10 100 10"
    expected_output = "-80"
    run_pie_test_case("../p03171.py", input_content, expected_output)


def test_problem_p03171_4():
    input_content = "1\n10"
    expected_output = "10"
    run_pie_test_case("../p03171.py", input_content, expected_output)


def test_problem_p03171_5():
    input_content = "6\n4 2 9 7 1 5"
    expected_output = "2"
    run_pie_test_case("../p03171.py", input_content, expected_output)
