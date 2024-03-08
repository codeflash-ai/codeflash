from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03578_0():
    input_content = "5\n3 1 4 1 5\n3\n5 4 3"
    expected_output = "YES"
    run_pie_test_case("../p03578.py", input_content, expected_output)


def test_problem_p03578_1():
    input_content = "5\n3 1 4 1 5\n3\n5 4 3"
    expected_output = "YES"
    run_pie_test_case("../p03578.py", input_content, expected_output)


def test_problem_p03578_2():
    input_content = "1\n800\n5\n100 100 100 100 100"
    expected_output = "NO"
    run_pie_test_case("../p03578.py", input_content, expected_output)


def test_problem_p03578_3():
    input_content = "15\n1 2 2 3 3 3 4 4 4 4 5 5 5 5 5\n9\n5 4 3 2 1 2 3 4 5"
    expected_output = "YES"
    run_pie_test_case("../p03578.py", input_content, expected_output)


def test_problem_p03578_4():
    input_content = "7\n100 200 500 700 1200 1600 2000\n6\n100 200 500 700 1600 1600"
    expected_output = "NO"
    run_pie_test_case("../p03578.py", input_content, expected_output)
