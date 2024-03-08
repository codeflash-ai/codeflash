from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02691_0():
    input_content = "6\n2 3 3 1 3 1"
    expected_output = "3"
    run_pie_test_case("../p02691.py", input_content, expected_output)


def test_problem_p02691_1():
    input_content = "6\n5 2 4 2 8 8"
    expected_output = "0"
    run_pie_test_case("../p02691.py", input_content, expected_output)


def test_problem_p02691_2():
    input_content = "6\n2 3 3 1 3 1"
    expected_output = "3"
    run_pie_test_case("../p02691.py", input_content, expected_output)


def test_problem_p02691_3():
    input_content = "32\n3 1 4 1 5 9 2 6 5 3 5 8 9 7 9 3 2 3 8 4 6 2 6 4 3 3 8 3 2 7 9 5"
    expected_output = "22"
    run_pie_test_case("../p02691.py", input_content, expected_output)
