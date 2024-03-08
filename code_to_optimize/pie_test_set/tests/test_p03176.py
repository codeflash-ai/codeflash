from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03176_0():
    input_content = "4\n3 1 4 2\n10 20 30 40"
    expected_output = "60"
    run_pie_test_case("../p03176.py", input_content, expected_output)


def test_problem_p03176_1():
    input_content = "5\n1 2 3 4 5\n1000000000 1000000000 1000000000 1000000000 1000000000"
    expected_output = "5000000000"
    run_pie_test_case("../p03176.py", input_content, expected_output)


def test_problem_p03176_2():
    input_content = "4\n3 1 4 2\n10 20 30 40"
    expected_output = "60"
    run_pie_test_case("../p03176.py", input_content, expected_output)


def test_problem_p03176_3():
    input_content = "1\n1\n10"
    expected_output = "10"
    run_pie_test_case("../p03176.py", input_content, expected_output)


def test_problem_p03176_4():
    input_content = "9\n4 2 5 8 3 6 1 7 9\n6 8 8 4 6 3 5 7 5"
    expected_output = "31"
    run_pie_test_case("../p03176.py", input_content, expected_output)
