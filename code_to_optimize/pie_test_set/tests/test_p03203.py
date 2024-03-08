from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03203_0():
    input_content = "3 3 1\n3 2"
    expected_output = "2"
    run_pie_test_case("../p03203.py", input_content, expected_output)


def test_problem_p03203_1():
    input_content = "3 3 1\n3 2"
    expected_output = "2"
    run_pie_test_case("../p03203.py", input_content, expected_output)


def test_problem_p03203_2():
    input_content = "100000 100000 0"
    expected_output = "100000"
    run_pie_test_case("../p03203.py", input_content, expected_output)


def test_problem_p03203_3():
    input_content = (
        "10 10 14\n4 3\n2 2\n7 3\n9 10\n7 7\n8 1\n10 10\n5 4\n3 4\n2 8\n6 4\n4 4\n5 8\n9 2"
    )
    expected_output = "6"
    run_pie_test_case("../p03203.py", input_content, expected_output)
