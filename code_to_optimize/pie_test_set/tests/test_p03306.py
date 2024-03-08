from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03306_0():
    input_content = "3 3\n1 2 3\n2 3 5\n1 3 4"
    expected_output = "1"
    run_pie_test_case("../p03306.py", input_content, expected_output)


def test_problem_p03306_1():
    input_content = "4 3\n1 2 6\n2 3 7\n3 4 5"
    expected_output = "3"
    run_pie_test_case("../p03306.py", input_content, expected_output)


def test_problem_p03306_2():
    input_content = "3 3\n1 2 3\n2 3 5\n1 3 4"
    expected_output = "1"
    run_pie_test_case("../p03306.py", input_content, expected_output)


def test_problem_p03306_3():
    input_content = (
        "8 7\n1 2 1000000000\n2 3 2\n3 4 1000000000\n4 5 2\n5 6 1000000000\n6 7 2\n7 8 1000000000"
    )
    expected_output = "0"
    run_pie_test_case("../p03306.py", input_content, expected_output)
