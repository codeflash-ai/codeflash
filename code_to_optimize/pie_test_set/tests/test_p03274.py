from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03274_0():
    input_content = "5 3\n-30 -10 10 20 50"
    expected_output = "40"
    run_pie_test_case("../p03274.py", input_content, expected_output)


def test_problem_p03274_1():
    input_content = "8 5\n-9 -7 -4 -3 1 2 3 4"
    expected_output = "10"
    run_pie_test_case("../p03274.py", input_content, expected_output)


def test_problem_p03274_2():
    input_content = "5 3\n-30 -10 10 20 50"
    expected_output = "40"
    run_pie_test_case("../p03274.py", input_content, expected_output)


def test_problem_p03274_3():
    input_content = "1 1\n0"
    expected_output = "0"
    run_pie_test_case("../p03274.py", input_content, expected_output)


def test_problem_p03274_4():
    input_content = "3 2\n10 20 30"
    expected_output = "20"
    run_pie_test_case("../p03274.py", input_content, expected_output)
