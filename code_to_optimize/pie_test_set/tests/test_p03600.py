from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03600_0():
    input_content = "3\n0 1 3\n1 0 2\n3 2 0"
    expected_output = "3"
    run_pie_test_case("../p03600.py", input_content, expected_output)


def test_problem_p03600_1():
    input_content = "3\n0 1000000000 1000000000\n1000000000 0 1000000000\n1000000000 1000000000 0"
    expected_output = "3000000000"
    run_pie_test_case("../p03600.py", input_content, expected_output)


def test_problem_p03600_2():
    input_content = "3\n0 1 3\n1 0 2\n3 2 0"
    expected_output = "3"
    run_pie_test_case("../p03600.py", input_content, expected_output)


def test_problem_p03600_3():
    input_content = "3\n0 1 3\n1 0 1\n3 1 0"
    expected_output = "-1"
    run_pie_test_case("../p03600.py", input_content, expected_output)


def test_problem_p03600_4():
    input_content = "5\n0 21 18 11 28\n21 0 13 10 26\n18 13 0 23 13\n11 10 23 0 17\n28 26 13 17 0"
    expected_output = "82"
    run_pie_test_case("../p03600.py", input_content, expected_output)
