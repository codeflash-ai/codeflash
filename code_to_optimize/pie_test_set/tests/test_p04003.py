from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04003_0():
    input_content = "3 3\n1 2 1\n2 3 1\n3 1 2"
    expected_output = "1"
    run_pie_test_case("../p04003.py", input_content, expected_output)


def test_problem_p04003_1():
    input_content = (
        "8 11\n1 3 1\n1 4 2\n2 3 1\n2 5 1\n3 4 3\n3 6 3\n3 7 3\n4 8 4\n5 6 1\n6 7 5\n7 8 5"
    )
    expected_output = "2"
    run_pie_test_case("../p04003.py", input_content, expected_output)


def test_problem_p04003_2():
    input_content = "2 0"
    expected_output = "-1"
    run_pie_test_case("../p04003.py", input_content, expected_output)


def test_problem_p04003_3():
    input_content = "3 3\n1 2 1\n2 3 1\n3 1 2"
    expected_output = "1"
    run_pie_test_case("../p04003.py", input_content, expected_output)
