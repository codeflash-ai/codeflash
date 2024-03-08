from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01719_0():
    input_content = "2 2 5\n3 2\n5 4"
    expected_output = "9"
    run_pie_test_case("../p01719.py", input_content, expected_output)


def test_problem_p01719_1():
    input_content = "1 2 5\n6\n10000"
    expected_output = "5"
    run_pie_test_case("../p01719.py", input_content, expected_output)


def test_problem_p01719_2():
    input_content = "2 3 5\n4 5\n6 3\n8 5"
    expected_output = "11"
    run_pie_test_case("../p01719.py", input_content, expected_output)


def test_problem_p01719_3():
    input_content = "3 3 10\n10 9 6\n8 7 3\n7 5 1"
    expected_output = "10"
    run_pie_test_case("../p01719.py", input_content, expected_output)


def test_problem_p01719_4():
    input_content = "2 2 5\n3 2\n5 4"
    expected_output = "9"
    run_pie_test_case("../p01719.py", input_content, expected_output)
