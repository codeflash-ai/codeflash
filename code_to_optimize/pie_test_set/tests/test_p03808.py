from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03808_0():
    input_content = "5\n4 5 1 2 3"
    expected_output = "YES"
    run_pie_test_case("../p03808.py", input_content, expected_output)


def test_problem_p03808_1():
    input_content = "4\n1 2 3 1"
    expected_output = "NO"
    run_pie_test_case("../p03808.py", input_content, expected_output)


def test_problem_p03808_2():
    input_content = "5\n6 9 12 10 8"
    expected_output = "YES"
    run_pie_test_case("../p03808.py", input_content, expected_output)


def test_problem_p03808_3():
    input_content = "5\n4 5 1 2 3"
    expected_output = "YES"
    run_pie_test_case("../p03808.py", input_content, expected_output)
