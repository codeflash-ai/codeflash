from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03959_0():
    input_content = "5\n1 3 3 3 3\n3 3 2 2 2"
    expected_output = "4"
    run_pie_test_case("../p03959.py", input_content, expected_output)


def test_problem_p03959_1():
    input_content = "5\n1 3 3 3 3\n3 3 2 2 2"
    expected_output = "4"
    run_pie_test_case("../p03959.py", input_content, expected_output)


def test_problem_p03959_2():
    input_content = "10\n1 3776 3776 8848 8848 8848 8848 8848 8848 8848\n8848 8848 8848 8848 8848 8848 8848 8848 3776 5"
    expected_output = "884111967"
    run_pie_test_case("../p03959.py", input_content, expected_output)


def test_problem_p03959_3():
    input_content = "1\n17\n17"
    expected_output = "1"
    run_pie_test_case("../p03959.py", input_content, expected_output)


def test_problem_p03959_4():
    input_content = "5\n1 1 1 2 2\n3 2 1 1 1"
    expected_output = "0"
    run_pie_test_case("../p03959.py", input_content, expected_output)
