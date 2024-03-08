from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02549_0():
    input_content = "5 2\n1 1\n3 4"
    expected_output = "4"
    run_pie_test_case("../p02549.py", input_content, expected_output)


def test_problem_p02549_1():
    input_content = "5 2\n3 3\n5 5"
    expected_output = "0"
    run_pie_test_case("../p02549.py", input_content, expected_output)


def test_problem_p02549_2():
    input_content = "5 1\n1 2"
    expected_output = "5"
    run_pie_test_case("../p02549.py", input_content, expected_output)


def test_problem_p02549_3():
    input_content = "60 3\n5 8\n1 3\n10 15"
    expected_output = "221823067"
    run_pie_test_case("../p02549.py", input_content, expected_output)


def test_problem_p02549_4():
    input_content = "5 2\n1 1\n3 4"
    expected_output = "4"
    run_pie_test_case("../p02549.py", input_content, expected_output)
