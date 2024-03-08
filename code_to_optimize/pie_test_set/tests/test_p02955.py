from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02955_0():
    input_content = "2 3\n8 20"
    expected_output = "7"
    run_pie_test_case("../p02955.py", input_content, expected_output)


def test_problem_p02955_1():
    input_content = "8 7\n1 7 5 6 8 2 6 5"
    expected_output = "5"
    run_pie_test_case("../p02955.py", input_content, expected_output)


def test_problem_p02955_2():
    input_content = "2 10\n3 5"
    expected_output = "8"
    run_pie_test_case("../p02955.py", input_content, expected_output)


def test_problem_p02955_3():
    input_content = "2 3\n8 20"
    expected_output = "7"
    run_pie_test_case("../p02955.py", input_content, expected_output)


def test_problem_p02955_4():
    input_content = "4 5\n10 1 2 22"
    expected_output = "7"
    run_pie_test_case("../p02955.py", input_content, expected_output)
